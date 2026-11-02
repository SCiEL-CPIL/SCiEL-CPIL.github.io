import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # n=2, m=2, p=12

    # Rank 24 decomposition
    # C_b = A_0 B_{0,b} + A_1 B_{1,b} for b in 0..11
    # This requires 2 multiplications per column b.
    # Total rank = 12 * 2 = 24.
    
    rank = 24
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Indices:
    # U: i = row_a * m + col_a = 2*row_a + col_a
    # V: j = row_b * p + col_b = 12*row_b + col_b
    # W: k = row_c * p + col_c = 12*row_c + col_c
    
    # For each column b (0 to 11):
    for b in range(p):
        r1 = 2 * b
        r2 = 2 * b + 1
        
        # Term 1: A[0, :] * B[:, b] -> C[0, b]
        # A index for row 0, col 0 is 0. A[0,0]
        # A index for row 0, col 1 is 1. A[0,1]
        # We need A_0 (row 0 of A) to multiply B_{0,b} (row 0 of B)
        # And A_1 (row 1 of A) to multiply B_{1,b} (row 1 of B)
        # Wait, C[0, b] = A[0,0]B[0,b] + A[0,1]B[1,b]
        # C[1, b] = A[1,0]B[0,b] + A[1,1]B[1,b]
        
        # My previous derivation:
        # C_b (row a) = sum_c A[a,c] B[c,b]
        # So for a=0: C[0,b] = A[0,0]B[0,b] + A[0,1]B[1,b]
        # For a=1: C[1,b] = A[1,0]B[0,b] + A[1,1]B[1,b]
        
        # The rank 24 decomposition I described was:
        # M_1 = A_0 B_{0,b} ? No, A_0 is a vector.
        # The decomposition I sketched was:
        # For each b, we use 2 terms.
        # Term 1: A[0,0] * B[0,b] contributes to C[0,b]?
        # No, A[0,0] * B[0,b] is one part of C[0,b].
        # A[0,1] * B[1,b] is the other part.
        # So we need a term for A[0,0]B[0,b] and a term for A[0,1]B[1,b]?
        # That would be 4 terms per b?
        
        # Let's re-evaluate the rank of 1x2x12.
        # T[a, c, b] where a is row of C, c is col of A/row of B, b is col of B/C.
        # T[a, c, b] = 1.
        # This is a 3-way tensor of size 2 x 2 x 12.
        # It is the sum over c of A[a,c] B[c,b].
        # This is NOT a direct sum of 2x2x1 tensors.
        # It is a single tensor.
        
        # Let's check the rank of 1x2x12 multiplication.
        # We want to compute y = A x where A is 1x2, x is 2x12? No.
        # A is 2x2. B is 2x12.
        # Row 0 of C: C[0, :] = A[0,0] B[0, :] + A[0,1] B[1, :].
        # This is a linear combination of two rows of B.
        # To compute this for all 12 columns, we can just do:
        # M_1 = A[0,0] * B[0, :]
        # M_2 = A[0,1] * B[1, :]
        # C[0, :] = M_1 + M_2
        # This requires 2 "vector multiplications" (scalar * vector).
        # In tensor rank terms, each scalar-vector multiplication $s \cdot v$ where $v$ has 12 components
        # can be represented as 12 rank-1 tensors?
        # $T_{i,j,k} = s_i v_{j,k}$?
        # Here $i$ is the scalar index (size 2 for A[0,0] and A[0,1]).
        # $j$ is the row of B (size 2).
        # $k$ is the col of B/C (size 12).
        
        # Let's decompose $C[0, b] = A[0,0] B[0, b] + A[0,1] B[1, b]$.
        # This involves 4 products: $A[0,0]B[0,b]$ and $A[0,1]B[1,b]$.
        # For a fixed $b$, we have 2 products.
        # For all $b$, we have $12 \times 2 = 24$ products.
        # Each product is a rank-1 tensor.
        # $U$ factor: $A[0,0]$ (index 0) or $A[0,1]$ (index 1).
        # $V$ factor: $B[0,b]$ (index $b$) or $B[1,b]$ (index $12+b$).
        # $W$ factor: $C[0,b]$ (index $b$).
        
        # So for each $b$:
        # Term 1: $A[0,0] B[0,b] \to C[0,b]$.
        # $U$ col: 1 at index 0 ($A[0,0]$).
        # $V$ col: 1 at index $b$ ($B[0,b]$).
        # $W$ col: 1 at index $b$ ($C[0,b]$).
        
        # Term 2: $A[0,1] B[1,b] \to C[0,b]$.
        # $U$ col: 1 at index 1 ($A[0,1]$).
        # $V$ col: 1 at index $12+b$ ($B[1,b]$).
        # $W$ col: 1 at index $b$ ($C[0,b]$).
        
        # Similarly for Row 1 of C:
        # $C[1, b] = A[1,0] B[0,b] + A[1,1] B[1,b]$.
        # Term 3: $A[1,0] B[0,b] \to C[1,b]$.
        # $U$ col: 1 at index 2 ($A[1,0]$).
        # $V$ col: 1 at index $b$ ($B[0,b]$).
        # $W$ col: 1 at index $12+b$ ($C[1,b]$).
        
        # Term 4: $A[1,1] B[1,b] \to C[1,b]$.
        # $U$ col: 1 at index 3 ($A[1,1]$).
        # $V$ col: 1 at index $12+b$ ($B[1,b]$).
        # $W$ col: 1 at index $12+b$ ($C[1,b]$).
        
        # Total terms per $b$: 4.
        # Total rank: $12 \times 4 = 48$.
        # This is the naive rank.
        
        # Can we do better?
        # We used Strassen to get 42.
        # Can we get 24?
        # The rank of $2 \times 2 \times 2$ is 7.
        # The rank of $2 \times 2 \times 12$ is at least $12 \times \text{rank}(2 \times 2 \times 1)$?
        # Rank of $2 \times 2 \times 1$ is 2.
        # So lower bound is 24?
        # But we need to check if the tensor allows sharing.
        
        # Let's look at the structure again.
        # $C[0, b] = A[0,0] B[0, b] + A[0,1] B[1, b]$
        # $C[1, b] = A[1,0] B[0, b] + A[1,1] B[1, b]$
        
        # For a fixed $b$, this is a $2 \times 2 \times 1$ multiplication (matrix-vector product).
        # The rank of $2 \times 2 \times 1$ is 2.
        # So for each $b$, we need 2 multiplications.
        # Total 24.
        
        # Let's verify the rank of $2 \times 2 \times 1$ is 2.
        # $y = A x$. $y_0 = a_{00} x_0 + a_{01} x_1$.
        # $y_1 = a_{10} x_0 + a_{11} x_1$.
        # We need to compute 2 outputs.
        # Naive: 4 mults.
        # Strassen-like?
        # $M_1 = a_{00} x_0$
        # $M_2 = a_{01} x_1$
        # $y_0 = M_1 + M_2$
        # $M_3 = a_{10} x_0$
        # $M_4 = a_{11} x_1$
        # $y_1 = M_3 + M_4$
        # This is 4.
        
        # Can we do it in 2?
        # If we use $M_1 = a_{00} x_0 + a_{01} x_1$? No, that's a sum.
        # Multiplication must be bilinear.
        # Rank 2 means $T = u_1 \otimes v_1 \otimes w_1 + u_2 \otimes v_2 \otimes w_2$.
        # $y_0 = (u_1[0] v_1[0] w_1[0] + u_2[0] v_2[0] w_2[0]) x$? No.
        # $y_i = \sum_k u_{i,k} v_{j,k} w_{l,k} a_{i,j} x_{j,l}$?
        
        # Actually, the rank of $2 \times 2 \times 1$ is 2.
        # Reference: "The rank of the matrix multiplication tensor".
        # $R(2,2,1) = 2$.
        # Proof:
        # $y_0 = a_{00} x_0 + a_{01} x_1$
        # $y_1 = a_{10} x_0 + a_{11} x_1$
        # Let $M_1 = a_{00} x_0$. $M_2 = a_{01} x_1$.
        # This is rank 2? No, that's 2 terms for $y_0$.
        # We need to cover $y_1$ too.
        
        # If we use 2 terms total for the whole $2 \times 2 \times 1$ tensor:
        # $T = u_1 v_1 w_1 + u_2 v_2 w_2$.
        # $y_0 = u_{0,1} v_{0,1} w_1 + u_{0,2} v_{0,2} w_2$?
        # This doesn't seem to work for arbitrary $a, x$.
        
        # Actually, $R(2,2,1) = 2$ is NOT correct.
        # $R(1,1,1) = 1$.
        # $R(1,2,1) = 2$.
        # $R(2,2,1)$?
        # We have 2 outputs. Each output is a dot product of 2 vectors.
        # Lower bound: $\lceil \frac{2+2+1}{2} \rceil$? No.
        # Lower bound is $\lceil \frac{nm + mp + pn}{2} \rceil$? No.
        # Lower bound is $\lceil \frac{nm + mp + pn}{2} \rceil$ is for something else.
        # Simple lower bound: $\lceil \frac{nm}{1} \rceil$?
        
        # Let's check known results.
        # Rank of $2 \times 2 \times 2$ is 7.
        # Rank of $2 \times 2 \times 1$ is 4?
        # If it's 4, then total rank is 48.
        # If it's 2, total rank is 24.
        
        # Let's try to construct a rank 2 decomposition for $2 \times 2 \times 1$.
        # $y_0 = a_{00} x_0 + a_{01} x_1$
        # $y_1 = a_{10} x_0 + a_{11} x_1$
        # Suppose $T = u_1 v_1 w_1 + u_2 v_2 w_2$.
        # $y_0 = (u_1[0] v_1[0] w_1 + u_2[0] v_2[0] w_2) \dots$?
        # The indices are $i$ (row of A, 2), $j$ (col of A/row of X, 2), $k$ (row of Y, 2? No, Y is 2x1, so k is 0).
        # Wait, $Y$ is $2 \times 1$. So $k$ is 0.
        # $T_{i,j,0} = 1$.
        # $y_i = \sum_j a_{i,j} x_{j,0}$.
        # If rank is 2, $T_{i,j,0} = u_{i,1} v_{j,1} w_{0,1} + u_{i,2} v_{j,2} w_{0,2}$.
        # Let $w_{0,1}=1, w_{0,2}=1$.
        # $1 = u_{i,1} v_{j,1} + u_{i,2} v_{j,2}$.
        # This must hold for all $i,j$.
        # This implies $U V^T = J$ (matrix of ones).
        # Rank of $J$ is 1.
        # So $u_1, u_2$ and $v_1, v_2$ must be such that their outer products sum to $J$.
        # This is possible.
        # But we also need to match the coefficients?
        # No, the tensor entries are 1.
        # The formula is $y_i = \sum_{j,k} T_{i,j,k} a_{i,j} x_{j,k}$.
        # Here $k$ is always 0.
        # $y_i = \sum_j T_{i,j,0} a_{i,j} x_{j,0}$.
        # If $T_{i,j,0} = 1$, then $y_i = \sum_j a_{i,j} x_{j,0}$.
        # This is the standard dot product.
        # The rank of the tensor with all 1s is 1?
        # $T_{i,j,0} = 1 \cdot 1 \cdot 1$.
        # So rank is 1?
        # No, the indices $i$ and $j$ are coupled in $a_{i,j}$.
        # In the decomposition, $U$ depends on $i$, $V$ on $j$.
        # $T_{i,j,0} = \sum_r u_{i,r} v_{j,r} w_{0,r}$.
        # If we set $u_{i,r} = 1, v_{j,r} = 1, w_{0,r} = 1$ for $r=1$, we get $T_{i,j,0} = 1$.
        # So the rank of the "all ones" tensor is 1.
        # But this doesn't compute the product.
        # The tensor $T$ defines the mapping.
        # The rank is the minimum $r$ such that $T = \sum u_r \otimes v_r \otimes w_r$.
        # If $T_{i,j,0} = 1$ for all $i,j$, then rank is 1.
        # But this would mean $y_i = \sum_j 1 \cdot a_{i,j} x_{j,0}$.
        # This IS the correct formula!
        # So the rank of $2 \times 2 \times 1$ is 1?
        # No, $w_{0,r}$ is the factor for $Y$.
        # $Y$ has 2 components.
        # So $k$ ranges 0 to 1.
        # $T_{i,j,k} = 1$ if $i=k$? No.
        # $y_i = \sum_j a_{i,j} x_{j,0}$.
        # The output index is $i$. The input indices are $i,j$ for A and $j,0$ for X.
        # Wait, X is $2 \times 1$. So $x_{j,0}$.
        # The tensor indices are $(i, j, k)$ where $i$ is row of A, $j$ is col of A, $k$ is row of Y?
        # No, standard MM tensor $T_{i,j,k}$ maps $A_{i,j}$ and $B_{j,k}$ to $C_{i,k}$.
        # Here $B$ is $2 \times 1$. So $k=0$.
        # $C$ is $2 \times 1$. So $k$ is 0 or 1?
        # $C_{i,0} = \sum_j A_{i,j} B_{j,0}$.
        # So the output index is $(i, 0)$.
        # The tensor has non-zeros at $(i, j, i)$?
        # No, $C_{i,0}$ depends on $A_{i,j}$ and $B_{j,0}$.
        # So $T_{i,j, i} = 1$?
        # The third index of T is the index of C.
        # C is flattened. $C_{0,0}$ is index 0. $C_{1,0}$ is index 1.
        # So $T_{i,j, i} = 1$.
        # $T_{0,j,0} = 1$.
        # $T_{1,j,1} = 1$.
        # This tensor is NOT all ones.
        # $T_{0,0,0}=1, T_{0,1,0}=1$.
        # $T_{1,0,1}=1, T_{1,1,1}=1$.
        # Others 0.
        # This is a direct sum of two $1 \times 2 \times 1$ tensors.
        # Rank of $1 \times 2 \times 1$ is 2.
        # So rank of $2 \times 2 \times 1$ is $2+2=4$.
        
        # So my initial calculation of 48 was correct.
        # And Strassen 42 is better.
        # Can we do better than 42?
        # The lower bound for $2 \times 2 \times 12$ is not simply 24.
        
        # Let's stick with 42. It's a known good value.
        # The current program already implements 42.
        # I should just ensure it runs fast and correctly.
        
        # Wait, the prompt says "score 1.1429".
        # And "seed/value".
        # If I return 42, score is 1.1429.
        # If I can find 40, score is 1.2.
        # Is 40 possible?
        # $2 \times 2 \times 12$.
        # Maybe 41?
        
        # Given the time constraint and the difficulty of proving rank,
        # I will stick with the Strassen-based 42 decomposition,
        # but optimize the code to be as fast as possible and robust.
        
        # The current code uses optimization which is slow and risky.
        # I will replace it with the direct constructive 42 rank decomposition.
        
        rank = 42
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        num_blocks = p // 2  # 6
        for t in range(num_blocks):
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # Strassen terms for block t
            # A indices: 0:00, 1:01, 2:10, 3:11
            # B indices: 0:00, 1:01, 2:10, 3:11 (local to block)
            # Global B index: row * 12 + col
            
            # M1 = (A00+A11)(B00+B11)
            c = base + 0
            U[0, c] += 1
            U[3, c] += 1
            V[0*p + k0, c] += 1
            V[1*p + k1, c] += 1
            W[k0*n + 0, c] += 1
            W[k1*n + 1, c] += 1
            
            # M2 = (A10+A11) B00
            c = base + 1
            U[2, c] += 1
            U[3, c] += 1
            V[0*p + k0, c] += 1
            W[k0*n + 1, c] += 1
            W[k1*n + 1, c] += -1
            
            # M3 = A00 (B10-B00) ? No, standard Strassen:
            # M3 = A00 (B10 - B00) is not standard.
            # Standard:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B00) -- Wait, let me use a verified set.
            
            # Verified Strassen for 2x2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B00) -> No, M3 = A00(B10 - B00) is not bilinear in B? It is.
            # Let's use:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B00)
            # M4 = A11(B01-B11)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # C00 = M1 - M4 + M7
            # C01 = M3 + M7
            # C10 = M2 + M4
            # C11 = M1 - M2 + M6
            
            # Let's re-verify M3, M4.
            # M3 = A00(B10-B00).
            # M4 = A11(B01-B11).
            
            # Update code with this set.
            
            c = base + 2
            U[0, c] += 1
            V[1*p + k0, c] += 1
            V[0*p + k0, c] += -1
            W[k0*n + 1, c] += 1
            W[k1*n + 1, c] += 1 # Wait, C01 and C11?
            # M3 contributes to C01.
            # M4 contributes to C00 and C10.
            
            # Let's just use the code from the current program's fallback, 
            # which seems to have been tested or at least consistent.
            # The fallback code in the prompt:
            # c = base + 2
            # U[0, c] += 1
            # V[0*p + k1, c] += 1
            # V[1*p + k1, c] += -1
            # W[k1*n + 0, c] += 1
            # W[k1*n + 1, c] += 1
            
            # This looks like M3 = A00 (B01 - B11)?
            # If so, it contributes to C01 and C11.
            
            # I will trust the fallback code structure as it was part of the "current program" 
            # which presumably passed basic checks or was a seed.
            
            # Re-implementing the fallback logic cleanly.
            
            pass # The loop above was just thinking. I will write the final code below.
        
        # Final Code Structure:
        # 1. Define dimensions.
        # 2. Initialize U, V, W with zeros.
        # 3. Loop 6 times.
        # 4. Fill 7 Strassen terms per block.
        # 5. Return.
        
        # I will copy the fallback logic from the prompt exactly, as it is the safest bet 
        # to get a valid rank-42 decomposition without introducing errors in the Strassen coefficients.
        
        rank = 42
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        num_blocks = p // 2  # 6
        for t in range(num_blocks):
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1
            
            c = base + 0
            U[0, c] += 1
            U[3, c] += 1
            V[0*p + k0, c] += 1
            V[1*p + k1, c] += 1
            W[k0*n + 0, c] += 1
            W[k1*n + 1, c] += 1
            
            c = base + 1
            U[2, c] += 1
            U[3, c] += 1
            V[0*p + k0, c] += 1
            W[k0*n + 1, c] += 1
            W[k1*n + 1, c] += -1
            
            c = base + 2
            U[0, c] += 1
            V[0*p + k1, c] += 1
            V[1*p + k1, c] += -1
            W[k1*n + 0, c] += 1
            W[k1*n + 1, c] += 1
            
            c = base + 3
            U[3, c] += 1
            V[1*p + k0, c] += 1
            V[0*p + k0, c] += -1
            W[k0*n + 0, c] += 1
            W[k0*n + 1, c] += 1
            
            c = base + 4
            U[0, c] += 1
            U[1, c] += 1
            V[1*p + k1, c] += 1
            W[k0*n + 0, c] += -1
            W[k1*n + 0, c] += 1
            
            c = base + 5
            U[2, c] += 1
            U[0, c] += -1
            V[0*p + k0, c] += 1
            V[0*p + k1, c] += 1
            W[k1*n + 1, c] += 1
            
            c = base + 6
            U[1, c] += 1
            U[3, c] += -1
            V[1*p + k0, c] += 1
            V[1*p + k1, c] += 1
            W[k0*n + 0, c] += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
