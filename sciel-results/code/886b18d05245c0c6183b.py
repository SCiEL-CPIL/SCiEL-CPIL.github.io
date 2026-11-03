import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = 3, 3, 3
    # Try rank 22 decomposition
    # 7 Strassen terms + 15 border terms
    
    # We will build the decomposition and verify it.
    # If verification fails, fall back to rank 27 (schoolbook) or 26.
    
    # Time guard
    start_time = time.time()
    
    # Rank 22 Candidate
    # U, V, W shapes: (9, rank), (9, rank), (9, rank)
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Let's represent the terms as (u_idx, v_idx, w_idx, coeff)
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    terms = []
    
    # Strassen Terms for 2x2 core (indices 0,1 for i,j; 0,1 for k)
    # A11=0, A12=1, A21=3, A22=4
    # B11=0, B12=3, B21=6, B22=9  (j*p+k: 0*3+0=0, 0*3+1=1? No. j is row of B? No. B is m x p. j=0..2, k=0..2. idx = j*p+k)
    # B[0,0]=0, B[0,1]=1, B[0,2]=2
    # B[1,0]=3, B[1,1]=4, B[1,2]=5
    # B[2,0]=6, B[2,1]=7, B[2,2]=8
    
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A[2,0]=6, A[2,1]=7, A[2,2]=8
    
    # W[k*n+i]
    # C[0,0]=0, C[0,1]=1, C[0,2]=2
    # C[1,0]=3, C[1,1]=4, C[1,2]=5
    # C[2,0]=6, C[2,1]=7, C[2,2]=8
    
    # Strassen M1..M7
    # M1 = (A11+A22)(B11+B22) -> A[0,0]+A[1,1] * B[0,0]+B[1,1]
    # u: A[0,0], A[1,1] -> idx 0, 4
    # v: B[0,0], B[1,1] -> idx 0, 4
    # This is not a single rank-1 term. It's a sum.
    # Strassen decomposition expresses C as linear combos of M1..M7.
    # So we need to expand M1..M7 into rank-1 terms?
    # No, Strassen IS the decomposition.
    # M1 is 1 term: (A11+A22) (B11+B22) C11...
    # But C11 is not just M1. C11 = M1 - M4 + M7 - M5.
    # So the rank-1 terms are the M's themselves?
    # No, the rank-1 terms are the products of linear forms.
    # The decomposition is:
    # C = sum_{k=1}^7 M_k * (linear combo of basis tensors)
    # So each M_k contributes 1 to the rank.
    # The M_k are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to U, V, W.
    # For M1:
    # u vector has 1 at A11 and A22.
    # v vector has 1 at B11 and B22.
    # w vector has 1 at C11, C12, C21, C22 with coefficients from Strassen.
    # Wait, W is the output matrix.
    # In the tensor decomposition $T = \sum u_r \otimes v_r \otimes w_r$.
    # $w_r$ is a vector in the output space.
    # For Strassen, $w_r$ is the vector of coefficients for the outputs.
    
    # Let's build the 7 Strassen terms.
    # Term 1: M1
    u1 = [0]*9; u1[0]=1; u1[4]=1
    v1 = [0]*9; v1[0]=1; v1[4]=1
    # Coeffs for C11, C12, C21, C22
    # C11 += M1
    # C12 -= M5 (from M5 def? No. C12 = M3+M5. C21=M2+M4. C22=M1-M2+M6-M3)
    # Let's get the standard Strassen output formulas:
    # C11 = M1 - M4 + M7 - M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 - M3
    
    # So for M1:
    w1 = [0]*9
    w1[0] = 1   # C11
    w1[4] = 1   # C22
    # M1 contributes to C11 and C22.
    
    # Term 2: M2
    u2 = [0]*9; u2[3]=1; u2[4]=1 # A21, A22
    v2 = [0]*9; v2[0]=1           # B11
    w2 = [0]*9
    w2[3] = 1   # C21
    w2[4] = -1  # C22
    
    # Term 3: M3
    u3 = [0]*9; u3[0]=1           # A11
    v3 = [0]*9; v3[1]=1; v3[4]=-1 # B12, B22
    w3 = [0]*9
    w3[1] = 1   # C12
    w3[4] = -1  # C22
    
    # Term 4: M4
    u4 = [0]*9; u4[4]=1           # A22
    v4 = [0]*9; v4[3]=1; v4[0]=-1 # B21, B11
    w4 = [0]*9
    w4[0] = -1  # C11
    w4[3] = 1   # C21
    
    # Term 5: M5
    u5 = [0]*9; u5[0]=1; u5[1]=1  # A11, A12
    v5 = [0]*9; v5[4]=1           # B22
    w5 = [0]*9
    w5[0] = -1  # C11
    w5[1] = 1   # C12
    
    # Term 6: M6
    u6 = [0]*9; u6[3]=1; u6[0]=-1 # A21, A11
    v6 = [0]*9; v6[0]=1; v6[1]=1  # B11, B12
    w6 = [0]*9
    w6[4] = 1   # C22
    
    # Term 7: M7
    u7 = [0]*9; u7[1]=1; u7[4]=-1 # A12, A22
    v7 = [0]*9; v7[3]=1; v7[4]=1  # B21, B22
    w7 = [0]*9
    w7[0] = 1   # C11
    
    # Now the 15 border terms.
    # They are simple products A_ij B_kl -> C_il.
    # Each is a rank-1 term with u having 1 at A, v having 1 at B, w having 1 at C.
    
    border_terms = []
    # C13
    border_terms.append((0, 2, 2, 1)) # A11 B13 -> C13. A[0,0]=0, B[0,2]=2, C[0,2]=2
    border_terms.append((1, 5, 2, 1)) # A12 B23 -> C13. A[0,1]=1, B[1,2]=5, C[0,2]=2
    border_terms.append((2, 8, 2, 1)) # A13 B33 -> C13. A[0,2]=2, B[2,2]=8, C[0,2]=2
    
    # C23
    border_terms.append((3, 2, 5, 1)) # A21 B13 -> C23. A[1,0]=3, B[0,2]=2, C[1,2]=5
    border_terms.append((4, 5, 5, 1)) # A22 B23 -> C23. A[1,1]=4, B[1,2]=5, C[1,2]=5
    border_terms.append((5, 8, 5, 1)) # A23 B33 -> C23. A[1,2]=5, B[2,2]=8, C[1,2]=5
    
    # C31
    border_terms.append((6, 0, 6, 1)) # A31 B11 -> C31. A[2,0]=6, B[0,0]=0, C[2,0]=6
    border_terms.append((7, 3, 6, 1)) # A32 B21 -> C31. A[2,1]=7, B[1,0]=3, C[2,0]=6
    border_terms.append((8, 6, 6, 1)) # A33 B31 -> C31. A[2,2]=8, B[2,0]=6, C[2,0]=6
    
    # C32
    border_terms.append((6, 1, 7, 1)) # A31 B12 -> C32. A[2,0]=6, B[0,1]=1, C[2,1]=7
    border_terms.append((7, 4, 7, 1)) # A32 B22 -> C32. A[2,1]=7, B[1,1]=4, C[2,1]=7
    border_terms.append((8, 7, 7, 1)) # A33 B32 -> C32. A[2,2]=8, B[2,1]=7, C[2,1]=7
    
    # C33
    border_terms.append((6, 2, 8, 1)) # A31 B13 -> C33. A[2,0]=6, B[0,2]=2, C[2,2]=8
    border_terms.append((7, 5, 8, 1)) # A32 B23 -> C33. A[2,1]=7, B[1,2]=5, C[2,2]=8
    border_terms.append((8, 8, 8, 1)) # A33 B33 -> C33. A[2,2]=8, B[2,2]=8, C[2,2]=8
    
    # Total rank 7 + 15 = 22.
    
    # Build U, V, W
    rank = 22
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Add Strassen terms
    for r, (u, v, w) in enumerate([(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4), (u5,v5,w5), (u6,v6,w6), (u7,v7,w7)]):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    # Add border terms
    for r, (ui, vi, wi, coeff) in enumerate(border_terms):
        idx = 7 + r
        U[ui, idx] = coeff
        V[vi, idx] = coeff
        W[wi, idx] = coeff
        
    # Verify
    # C = U^T A V? No.
    # T = sum U[:,k] A V[:,k] W[:,k]^T?
    # Standard: C[i,k] = sum_j A[i,j] B[j,k].
    # Decomposition: C = sum_{r} (U_r^T A) (V_r^T B) W_r^T?
    # Let's check the convention.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # C[i,k] = sum_j sum_r U[i*m+j, r] A[i,j] V[j*p+k, r] B[j,k] W[k*n+i, r] ?
    # No, the tensor is T[i,j,k,l] = delta_{j,k} delta_{i,l}? No.
    # T_{i,j,k} = A_{i,j} B_{j,k} -> C_{i,k}.
    # Decomposition: C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_j V_{j,k,r} B_{j,k}) W_{k,i,r}?
    # Actually, the term is $u_r \otimes v_r \otimes w_r$.
    # $C = \sum_r (u_r^T A) (v_r^T B) w_r^T$? No.
    # $C_{i,k} = \sum_r w_{k,i,r} (\sum_j u_{i,j,r} A_{i,j}) (\sum_j v_{j,k,r} B_{j,k})$?
    # This looks like the product of two scalars times w.
    # Let's verify with schoolbook.
    # U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
    # Term r corresponds to (i,j,k).
    # Sum over r: sum_{i,j,k} A_{i,j} B_{j,k} delta_{i,i} delta_{k,k}?
    # If U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i).
    # Then term is $A_{i,j} B_{j,k} \cdot 1$.
    # Sum over all (i,j,k).
    # This gives $\sum_j A_{i,j} B_{j,k}$. Correct.
    
    # So we need to verify:
    # C_calc[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Let's test with random A, B
    A_test = np.random.rand(3,3)
    B_test = np.random.rand(3,3)
    C_expected = A_test @ B_test
    
    C_calc = np.zeros((3,3))
    for r in range(rank):
        # Compute scalar parts
        # sum_j U[i*m+j, r] A[i,j]
        # This is not a simple matrix product.
        # For fixed r, let u_r be a 3x3 matrix.
        # term_r_A[i] = sum_j U[i*m+j, r] A[i,j]
        # term_r_B[k] = sum_j V[j*p+k, r] B[j,k]
        
        U_r = U[:, r].reshape(3,3)
        V_r = V[:, r].reshape(3,3)
        W_r = W[:, r].reshape(3,3) # W[k*n+i] -> W_r[k,i]
        
        # term_A[i] = sum_j U_r[i,j] A[i,j]
        term_A = np.sum(U_r * A_test, axis=1)
        # term_B[k] = sum_j V_r[j,k] B[j,k]
        term_B = np.sum(V_r * B_test, axis=0)
        
        # C_calc[i,k] += W_r[k,i] * term_A[i] * term_B[k]
        # W_r[k,i] is W[k*n+i]
        # term_A[i] is scalar for row i
        # term_B[k] is scalar for col k
        
        # Outer product of term_A and term_B?
        # C[i,k] = sum_r W_r[k,i] * term_A[i] * term_B[k]
        
        # Vectorize:
        # C += W_r * np.outer(term_A, term_B) ?
        # W_r[k,i] multiplies term_A[i] * term_B[k].
        # Yes.
        
        C_calc += W_r * np.outer(term_A, term_B)
        
    if np.allclose(C_calc, C_expected):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Rank 27
        rank = 27
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
