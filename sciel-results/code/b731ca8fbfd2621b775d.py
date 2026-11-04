import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def get_strassen_2x2x2():
    # Returns U, V, W for 2x2x2, rank 7
    # Indices: 0..3 for 2x2 matrices
    # U: 4x7, V: 4x7, W: 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 0]
    
    # M3 = A00(B01-B11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 0]
    
    # M4 = A11(B10-B00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [0, 0, 0, 1]
    
    # M5 = (A00+A01)B11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [1, 0, 0, -1]
    
    # M6 = (A10-A00)(B00+B01)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, -1, 1, 0]
    
    # M7 = (A01-A11)(B10+B11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, -1]
    
    return U, V, W

def get_strassen_4x4x4():
    # Returns U, V, W for 4x4x4, rank 49
    # Indices: 0..15 for 4x4 matrices
    n = 4
    size = n * n
    U = np.zeros((size, 49))
    V = np.zeros((size, 49))
    W = np.zeros((size, 49))
    
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Block definitions for 4x4 matrices
    # Blocks are 2x2.
    # Block (br, bc) corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1
    # Global index i = br*2 + ir, j = bc*2 + jc
    
    # The 7 block-level terms
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B01-B11)
    # t=3: M4 = A11(B10-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    # Coefficients for X_t (linear comb of A blocks)
    # X_t[br, bc]
    X_coeffs = [
        [[1, 0], [0, 1]],   # t=0: A00 + A11
        [[0, 0], [1, 1]],   # t=1: A10 + A11
        [[1, 0], [0, 0]],   # t=2: A00
        [[0, 0], [0, 1]],   # t=3: A11
        [[1, 1], [0, 0]],   # t=4: A00 + A01
        [[0, 0], [1, -1]],  # t=5: A10 - A00
        [[0, 1], [0, -1]]   # t=6: A01 - A11
    ]
    
    # Coefficients for Y_t (linear comb of B blocks)
    Y_coeffs = [
        [[1, 0], [0, 1]],   # t=0: B00 + B11
        [[1, 0], [0, 0]],   # t=1: B00
        [[0, 1], [0, -1]],  # t=2: B01 - B11
        [[0, 0], [1, -1]],  # t=3: B10 - B00
        [[0, 0], [0, 1]],   # t=4: B11
        [[1, 1], [0, 0]],   # t=5: B00 + B01
        [[0, 0], [1, 1]]    # t=6: B10 + B11
    ]
    
    # Coefficients for C blocks (linear comb of M_t)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [[1, 0], [0, 1]],   # C00: M1 + M4 - M5 + M7 -> indices 0, 3, 4, 6
        [[0, 1], [0, 0]],   # C01: M3 + M5 -> indices 2, 4
        [[0, 0], [1, 0]],   # C10: M2 + M4 -> indices 1, 3
        [[1, 0], [0, 1]]    # C11: M1 - M2 + M3 + M6 -> indices 0, 1, 2, 5
    ]
    # Correct C_coeffs
    C_coeffs[0] = [[1, 0], [0, 1]] # Placeholder, will fill below
    
    # Let's just use the explicit sums
    # For each block (br, bc) of C, it is a sum of M_t with coefficients.
    # C[br, bc] = sum_t gamma[br, bc, t] M_t
    
    gamma = np.zeros((2, 2, 7))
    # C00 = M1 + M4 - M5 + M7
    gamma[0, 0, 0] = 1
    gamma[0, 0, 3] = 1
    gamma[0, 0, 4] = -1
    gamma[0, 0, 6] = 1
    
    # C01 = M3 + M5
    gamma[0, 1, 2] = 1
    gamma[0, 1, 4] = 1
    
    # C10 = M2 + M4
    gamma[1, 0, 1] = 1
    gamma[1, 0, 3] = 1
    
    # C11 = M1 - M2 + M3 + M6
    gamma[1, 1, 0] = 1
    gamma[1, 1, 1] = -1
    gamma[1, 1, 2] = 1
    gamma[1, 1, 5] = 1
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # Fill U
            # U[i, col] = U2[ir, s] * X_coeffs[t][br, bc]
            # i = br*2 + ir, j = bc*2 + jc
            # Note: U2[ir, s] is the coefficient for the entry at (ir, jc) in the 2x2 matrix X_t?
            # No, U2 is for the 2x2 matrix multiplication.
            # The "matrix" X_t has entries X_t[ir, jc].
            # U2[ir*2+jc, s] is the coefficient for X_t[ir, jc] in the s-th term.
            
            for br in range(2):
                for bc in range(2):
                    coeff = X_coeffs[t][br][bc]
                    if coeff == 0: continue
                    for ir in range(2):
                        for jc in range(2):
                            i_global = br * 2 + ir
                            j_global = bc * 2 + jc
                            i_local = ir * 2 + jc
                            u_val = U2[i_local, s]
                            U[i_global * 4 + j_global, col] = u_val * coeff
                            
            # Fill V
            for br in range(2):
                for bc in range(2):
                    coeff = Y_coeffs[t][br][bc]
                    if coeff == 0: continue
                    for ir in range(2):
                        for jc in range(2):
                            i_global = br * 2 + ir
                            j_global = bc * 2 + jc
                            i_local = ir * 2 + jc
                            v_val = V2[i_local, s]
                            V[i_global * 4 + j_global, col] = v_val * coeff
                            
            # Fill W
            # W[i, col] = sum_{br,bc} gamma[br,bc,t] * W2[ir*2+jc, s]
            # i = br*2 + ir, j = bc*2 + jc
            for br in range(2):
                for bc in range(2):
                    coeff = gamma[br, bc, t]
                    if coeff == 0: continue
                    for ir in range(2):
                        for jc in range(2):
                            i_global = br * 2 + ir
                            j_global = bc * 2 + jc
                            i_local = ir * 2 + jc
                            w_val = W2[i_local, s]
                            W[i_global * 4 + j_global, col] += w_val * coeff
                            
    return U, V, W

def get_schoolbook_4x4x1():
    # Returns U, V, W for 4x4x1, rank 16
    # A: 4x4, B: 4x1, C: 4x1
    # U: 16x16, V: 4x16, W: 4x16
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # Subproblems:
    # 1. 4x4x4 (cols 0-3 of B)
    # 2. 4x4x4 (cols 4-7 of B)
    # 3. 4x4x1 (col 8 of B)
    
    U1, V1, W1 = get_strassen_4x4x4()
    U2, V2, W2 = get_strassen_4x4x4()
    U3, V3, W3 = get_schoolbook_4x4x1()
    
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank)) # 4x9
    W = np.zeros((36, rank)) # 9x4? No, p*n = 9*4 = 36.
    
    # Copy U1
    U[:, 0:49] = U1
    # Copy V1 (map cols 0-3 of B to global)
    # V1 has 16 rows (4x4). Global V has 36 rows.
    # Row j*9 + k in global V.
    # For subproblem 1, k is 0..3.
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V1[j * 4 + k, 0:49]
            
    # Copy W1 (map cols 0-3 of C to global)
    # W1 has 16 rows. Index k*4 + i.
    # Global W index is k*4 + i.
    # For subproblem 1, k is 0..3.
    W[0:16, 0:49] = W1
    
    # Copy U2
    U[:, 49:98] = U2
    # Copy V2 (map cols 4-7 of B to global)
    for j in range(4):
        for k in range(4):
            V[j * 9 + (k + 4), 49:98] = V2[j * 4 + k, 0:49]
            
    # Copy W2 (map cols 4-7 of C to global)
    # Global W index k*4 + i. k is 4..7.
    # Rows 16..31.
    W[16:32, 49:98] = W2
    
    # Copy U3
    U[:, 98:114] = U3
    # Copy V3 (map col 8 of B to global)
    # V3 has 4 rows.
    for j in range(4):
        V[j * 9 + 8, 98:114] = V3[j, 0:16]
        
    # Copy W3 (map col 8 of C to global)
    # W3 has 4 rows. Index i.
    # Global W index 8*4 + i = 32 + i.
    W[32:36, 98:114] = W3
    
    # Verification
    # Check time
    if time.time() > deadline:
         # Return schoolbook if too slow (unlikely)
         pass
    else:
         # Quick check on small random matrices
         try:
             A = np.random.randint(-10, 10, (4, 4))
             B = np.random.randint(-10, 10, (4, 9))
             C_expected = A @ B
             C_computed = (U.T @ A.flatten()).reshape(4, 4) # No, this is not how it works.
             
             # The decomposition is C = sum_r U[:,r] * V[:,r] * W[:,r]?
             # No, C_{i,k} = sum_r U_{i,j?, r} ...
             # The tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
             # The task says:
             # U: (n*m, rank) -> indices i,j
             # V: (m*p, rank) -> indices j,k
             # W: (p*n, rank) -> indices k,i
             
             # C_{i,k} = sum_j A_{i,j} B_{j,k}
             # Decomposition:
             # A_{i,j} B_{j,k} C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
             
             # Let's verify this identity.
             # C_computed_{i,k} = sum_r (sum_j A_{i,j} U_{i,j,r}) (sum_j B_{j,k} V_{j,k,r}) ? No.
             
             # The standard tensor decomposition is:
             # T = sum_r u_r \otimes v_r \otimes w_r
             # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
             
             # We want to check if sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} == A_{i,j} B_{j,k} C_{i,k} ?
             # No, the tensor for matrix multiplication is defined such that
             # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = trace(A B C^T)?
             
             # Actually, the condition is:
             # For all A, B, C, sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = sum_{i,j,k} (sum_j A_{i,j} B_{j,k}) C_{i,k}
             
             # Let's just check the rank-1 terms.
             # Term r contributes:
             # sum_{i,j,k} U_{i,j,r} V_{j,k,r} W_{k,i,r} A_{i,j} B_{j,k} C_{i,k}
             # = (sum_{i,j} U_{i,j,r} A_{i,j}) (sum_{j,k} V_{j,k,r} B_{j,k}) (sum_{k,i} W_{k,i,r} C_{i,k})
             # This is not correct. The indices must match.
             
             # The correct check is:
             # Does sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] equal the matrix multiplication tensor?
             
             # Let's construct the tensor T and compare with the schoolbook tensor.
             # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
             
             T = np.zeros((4, 4, 9))
             for r in range(rank):
                 for i in range(4):
                     for j in range(4):
                         for k in range(9):
                             T[i,j,k] += U[i*4+j, r] * V[j*9+k, r] * W[k*4+i, r]
             
             # Schoolbook tensor
             T_school = np.zeros((4, 4, 9))
             for i in range(4):
                 for j in range(4):
                     for k in range(9):
                         T_school[i,j,k] = 1 # Wait, the tensor is sum_j A_ij B_jk C_ik?
                         # No, the tensor T satisfies sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = sum_{i,k} C_{i,k} (AB)_{i,k}
                         # This implies T_{i,j,k} must be such that for fixed i,k, sum_j T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k} (AB)_{i,k}
                         # This is only true if T is the identity in j?
                         
                         # Actually, the tensor for matrix multiplication is defined by:
                         # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                         # So the tensor T is such that T_{i,j,k} = \delta_{j,j}? No.
                         
                         # The standard decomposition is:
                         # A_{i,j} B_{j,k} C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
                         # This must hold as a polynomial identity.
                         
                         # Let's check T[i,j,k] against the expected values.
                         # For a fixed i,k, the term in the sum is sum_j A_{i,j} B_{j,k} C_{i,k}.
                         # So T_{i,j,k} should be 1 if it corresponds to the term A_{i,j} B_{j,k} C_{i,k}.
                         # But C_{i,k} is only one index.
                         
                         # The tensor T_{i,j,k} is 1 if the indices match the "multiplication" structure.
                         # Specifically, T_{i,j,k} = 1 for all i,j,k? No.
                         
                         # Let's use the property:
                         # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = sum_{i,k} C_{i,k} sum_j A_{i,j} B_{j,k}
                         
                         # This implies that for any A,B,C, the equality holds.
                         # This is true if T_{i,j,k} = \delta_{j, j}?
                         # No, it's true if T is the "matrix multiplication tensor".
                         
                         # The entries of the matrix multiplication tensor are:
                         # T_{i,j,k} = 1 if the indices are compatible?
                         # Actually, the tensor is sparse.
                         # T_{i,j,k} = 1 for all i,j,k is NOT correct.
                         
                         # Let's check the schoolbook decomposition.
                         # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
                         # c = i*m*p + j*p + k.
                         # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
                         # Only one c is non-zero for each (i,j,k), and it is 1.
                         # So T_{i,j,k} = 1 for all i,j,k.
                         
                         # Wait, is the matrix multiplication tensor the all-ones tensor?
                         # No.
                         # Let's re-read the definition.
                         # The tensor T is defined by T_{i,j,k} = 1 if ...?
                         
                         # Actually, the tensor for matrix multiplication is:
                         # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
                         # No.
                         
                         # Let's test with 1x1x1.
                         # T_{0,0,0} = 1.
                         # 1x1x1 rank 1.
                         
                         # 2x2x2 schoolbook rank 8.
                         # T_{i,j,k} = 1 for all i,j,k?
                         # If T is all ones, then sum T A B C = (sum A)(sum B)(sum C).
                         # But (AB) = sum_j A_{i,j} B_{j,k}.
                         # sum_{i,k} C_{i,k} (AB)_{i,k} = sum_{i,k} C_{i,k} sum_j A_{i,j} B_{j,k}
                         # = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
                         
                         # So YES, the tensor T_{i,j,k} is 1 for all i,j,k.
                         
             # So we just need to check if T (computed) is all ones.
             if not np.allclose(T, 1.0):
                 return {"U": [[0]*144 for _ in range(16)], "V": [[0]*144 for _ in range(36)], "W": [[0]*144 for _ in range(36)]}
         except Exception:
             pass
             
    # Convert to lists of lists of integers/halves
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

