import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s margin
    
    n, m, p = DIMENSIONS
    # Split m=4 into 2 blocks of 2
    # Split p=6 into 3 blocks of 2
    # 2 * 3 = 6 blocks of 2x2x2
    # Each 2x2x2 has rank 7
    # Total rank 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen matrices for 2x2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices for U_S, V_S: row = i*2 + j
    # Indices for W_S: row = k*2 + i
    
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, -1, 1, 0, 0, -1] # A22
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1], # C11 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],  # C21 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],  # C12 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # C22 (k=1, i=1)
    ]
    
    # Verify W_S logic
    # C11 = M1 + M4 - M5 + M7
    # M1(0): 1, M4(3): 1, M5(4): -1, M7(6): 1. Sum = 2?
    # Let's re-verify the Strassen formulas.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check C11 with these W_S values.
    # W_S[0] is C11.
    # s=0 (M1): 1
    # s=3 (M4): 1
    # s=4 (M5): -1
    # s=6 (M7): 1
    # Sum = 2. This is wrong. It should be 1 for the coefficient of the product?
    # No, the tensor decomposition is linear.
    # The entry $T_{ijk}$ is $\sum_r U_{ir} V_{jr} W_{kr}$.
    # Here $U_{i,j}$ corresponds to $A_{ij}$, $V_{j,k}$ to $B_{jk}$, $W_{k,i}$ to $C_{ik}$?
    # The trace convention is $W[k \cdot n + i, r]$.
    # So $C_{ik} = \sum_r U_{i, \cdot, r} V_{\cdot, k, r} W_{k, i, r}$.
    # My indices:
    # $U$ row $i \cdot m + j$.
    # $V$ row $j \cdot p + k$.
    # $W$ row $k \cdot n + i$.
    #
    # For 2x2x2:
    # $U_S[i \cdot 2 + j, r]$.
    # $V_S[j \cdot 2 + k, r]$.
    # $W_S[k \cdot 2 + i, r]$.
    #
    # Let's check C11 (i=0, k=0).
    # $C_{00} = \sum_r U_S[0 \cdot 2 + 0, r] V_S[0 \cdot 2 + 0, r] W_S[0 \cdot 2 + 0, r]$
    # $= \sum_r U_S[0, r] V_S[0, r] W_S[0, r]$.
    # This must equal $A_{00} B_{00}$.
    # Wait, matrix multiplication $C = AB$. $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # For 2x2x2, $C_{00} = A_{00}B_{00} + A_{01}B_{10}$.
    #
    # Let's check if my Strassen matrices produce this.
    # r=0 (M1): U[0]=1, V[0]=1, W[0]=1. Prod=1.
    # r=1 (M2): U[0]=0, V[0]=1, W[0]=0. Prod=0.
    # r=2 (M3): U[0]=1, V[0]=0, W[0]=0. Prod=0.
    # r=3 (M4): U[0]=0, V[0]=-1, W[0]=1. Prod=0.
    # r=4 (M5): U[0]=1, V[0]=0, W[0]=-1. Prod=0.
    # r=5 (M6): U[0]=-1, V[0]=1, W[0]=0. Prod=0.
    # r=6 (M7): U[0]=0, V[0]=0, W[0]=1. Prod=0.
    # Sum = 1. This gives $A_{00}B_{00}$ term.
    #
    # Now check $A_{01}B_{10}$ term.
    # This comes from $j=1$ in the sum $\sum_j$.
    # But the tensor contraction sums over $r$, not $j$.
    # The $j$ index is internal to $U$ and $V$? No.
    # $U$ has indices $i, j$. $V$ has indices $j, k$.
    # The contraction is $\sum_r U[i, j, r] V[j, k, r] W[k, i, r]$?
    # No, the standard decomposition is:
    # $T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}$?
    # No, matrix multiplication tensor $T_{i, j, k}$ where $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The tensor is indexed by $(i, j, k)$?
    # The prompt says $U$ shape $(n \cdot m, \text{rank})$.
    # $U$ row index $i \cdot m + j$.
    # $V$ row index $j \cdot p + k$.
    # $W$ row index $k \cdot n + i$.
    # This implies the contraction is over the row indices of U and V matching?
    # No, $U$ row is $(i, j)$, $V$ row is $(j, k)$.
    # The shared index is $j$?
    # If we sum over $r$, we get a scalar.
    # The matrix multiplication is a bilinear map.
    # $C_{ik} = \sum_{j=0}^{m-1} A_{ij} B_{jk}$.
    # In tensor form, $T_{i, j, k} = \delta_{jk}$? No.
    # The tensor of matrix multiplication has indices $(i, j, k)$?
    # Usually $T_{i, j, k} = 1$ if $j$ matches?
    # Let's look at the schoolbook code.
    # $U[i \cdot m + j, c] = 1$.
    # $V[j \cdot p + k, c] = 1$.
    # $W[k \cdot n + i, c] = 1$.
    # $c$ corresponds to $(i, j, k)$.
    # So for each $(i, j, k)$, we have a rank-1 term.
    # $U_{(i, j), c} = 1$.
    # $V_{(j, k), c} = 1$.
    # $W_{(k, i), c} = 1$.
    # The product is $1 \cdot 1 \cdot 1 = 1$.
    # This term contributes to $C_{ik}$?
    # How does the evaluator reconstruct $C$?
    # Presumably $C_{ik} = \sum_r \sum_j U_{i, j, r} V_{j, k, r} W_{k, i, r}$?
    # No, the indices $i, j$ in $U$ are flattened.
    # The contraction must recover $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # If the evaluator computes $C_{ik} = \sum_r (\sum_j U_{i, j, r} A_{ij}) (\sum_k V_{j, k, r} B_{jk}) W_{k, i, r}$?
    # No, that's not how it works.
    # The standard definition:
    # $C_{ik} = \sum_r U_{i, \cdot, r} V_{\cdot, k, r} W_{k, i, r}$ is wrong because of $j$.
    # The rank-1 decomposition of the multiplication tensor $T_{i, j, k}$ (where $T_{i, j, k}=1$ if we are computing $C_{ik}$ using $A_{ij} B_{jk}$? No).
    # The multiplication tensor $\mathcal{M}$ is defined by $\mathcal{M}(A, B, C^*) = \sum_{i,j,k} A_{ij} B_{jk} C^*_{ik}$.
    # A rank-1 decomposition is $\sum_r U_r(A) V_r(B) W_r(C^*)$.
    # $U_r$ is linear in $A$, $V_r$ linear in $B$, $W_r$ linear in $C^*$.
    # $U_r(A) = \sum_{i,j} U_{i,j,r} A_{ij}$.
    # $V_r(B) = \sum_{j,k} V_{j,k,r} B_{jk}$.
    # $W_r(C^*) = \sum_{i,k} W_{i,k,r} C^*_{ik}$.
    # Note: Prompt says $W[k \cdot n + i, r]$. So $W_{k, i, r}$.
    # Then $\sum_r U_r(A) V_r(B) W_r(C^*) = \sum_{i,j,k} A_{ij} B_{jk} C^*_{ik} (\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r})$.
    # For this to equal $\sum_{i,j,k} A_{ij} B_{jk} C^*_{ik}$, we need $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$ if the indices match the multiplication?
    # No, the multiplication is $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So the tensor is $\sum_{i,k} C_{ik} (\sum_j A_{ij} B_{jk})$.
    # In terms of $C^*$, it is $\sum_{i,j,k} A_{ij} B_{jk} C^*_{ik}$? No, $C^*_{ik}$ pairs with $C_{ik}$.
    # So yes, the coefficient of $A_{ij} B_{jk} C^*_{ik}$ in the expansion must be 1 if $j$ is the intermediate index?
    # Actually, the term is $A_{ij} B_{jk} C^*_{ik}$.
    # So we need $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$ for all valid $i,j,k$?
    # No, that would imply $T_{i,j,k} = 1$ for all $i,j,k$.
    # But $T_{i,j,k}$ is only non-zero if the indices correspond to a valid multiplication term.
    # For $n=2, m=4, p=6$, all $i \in 0..1, j \in 0..3, k \in 0..5$ are valid.
    # So yes, $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$ must be 1 for all $i,j,k$.
    #
    # Let's check Strassen again.
    # For 2x2x2, does $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$?
    # i=0, j=0, k=0.
    # r=0: 1*1*1 = 1.
    # r=1: 0*1*0 = 0.
    # r=2: 1*0*0 = 0.
    # r=3: 0*(-1)*1 = 0.
    # r=4: 1*0*(-1) = 0.
    # r=5: -1*1*0 = 0.
    # r=6: 0*0*1 = 0.
    # Sum = 1. Correct.
    #
    # i=0, j=0, k=1.
    # r=0: 1*1*0 (W[2] for k=1,i=0 is W_S[2]=0) = 0.
    # r=1: 0*1*0 = 0.
    # r=2: 1*1*1 (W_S[2]=1) = 1.
    # r=3: 0*(-1)*0 = 0.
    # r=4: 1*0*1 (W_S[2]=1) = 0.
    # r=5: -1*1*0 = 0.
    # r=6: 0*0*0 = 0.
    # Sum = 1. Correct.
    #
    # It seems my Strassen matrices are correct.
    
    # Construct full matrices
    col = 0
    for I in range(2): # Block row of A (0, 1)
        for J in range(3): # Block col of B (0, 1, 2)
            for s in range(7):
                for i_l in range(2): # local i
                    for j_l in range(2): # local j
                        # U index
                        u_row = i_l * 4 + (2 * I + j_l)
                        val = U_S[i_l * 2 + j_l][s]
                        U[u_row][col] += val
                        
                        # V index
                        v_row = (2 * I + j_l) * 6 + (2 * J + (s % 2)) # Wait, V_S depends on j_l, k_l.
                        # V_S index is j_l * 2 + k_l.
                        # We need to sum over k_l?
                        # No, V_S is a matrix. V_S[j_l * 2 + k_l, s].
                        # We must fill V for all k_l.
                        pass
                
                # Better to loop over all local indices for U, V, W
                for i_l in range(2):
                    for j_l in range(2):
                        u_row = i_l * 4 + (2 * I + j_l)
                        U[u_row][col] += U_S[i_l * 2 + j_l][s]
                        
                        # V needs j_l, k_l
                        for k_l in range(2):
                            v_row = (2 * I + j_l) * 6 + (2 * J + k_l)
                            V[v_row][col] += V_S[j_l * 2 + k_l][s]
                            
                            # W needs k_l, i_l
                            w_row = (2 * J + k_l) * 2 + i_l
                            W[w_row][col] += W_S[k_l * 2 + i_l][s]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}

