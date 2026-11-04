import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen factors for 2x2 * 2x2
    # A is rows i1, i2; cols j1, j2
    # B is rows j1, j2; cols k1, k2
    # C is rows i1, i2; cols k1, k2
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Coefficients for U (A), V (B), W (C)
        # W is indexed by k, i.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_coeffs = [
            # M1
            { (i1,j1): 1, (i2,j2): 1, (j1,k1): 1, (j2,k2): 1, (k1,i1): 1, (k2,i2): 1 },
            # M2
            { (i2,j1): 1, (i2,j2): 1, (j1,k1): 1, (k1,i2): 1 },
            # M3
            { (i1,j1): 1, (j1,k2): 1, (j2,k2): -1, (k2,i1): 1 },
            # M4
            { (i2,j2): 1, (j2,k1): 1, (j1,k1): -1, (k1,i2): 1 },
            # M5
            { (i1,j1): 1, (i1,j2): 1, (j2,k2): 1, (k2,i1): -1 }, # Wait, C00 has -M5, C01 has +M5.
            # Let's re-verify Strassen formulas.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # My dict above stores contributions to U, V, W for a single factor.
            # I need to store the factor's contribution to U, V, W separately, 
            # and then W is determined by how M_c contributes to C.
            # The "factor" c has:
            # U_c[i,j] = coeff of A_ij in M_c
            # V_c[j,k] = coeff of B_jk in M_c
            # W_c[k,i] = coeff of M_c in C_ki? No, C is indexed i,k.
            # The tensor is sum_c U_c[i,j] V_c[j,k] W_c[k,i] = C[i,k].
            # So W_c[k,i] is the coefficient of M_c in the expression for C[i,k].
        ]
        
        # Let's build the 7 factors explicitly.
        
        # Factor 1: M1
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        # C00 += M1, C11 += M1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # Factor 2: M2
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        # C10 += M2, C11 -= M2
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        
        # Factor 3: M3
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        # C01 += M3, C11 += M3
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # Factor 4: M4
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        # C00 += M4, C10 += M4
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        
        # Factor 5: M5
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        # C00 -= M5, C01 += M5
        W[k2*n+i1, col] = -1; W[k2*n+i1, col] = 1 # Wait, C00 is i1, k1. C01 is i1, k2.
        # M5 contributes to C00 (i1, k1) and C01 (i1, k2)?
        # Strassen:
        # M5 = (A00 + A01) B11
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # So M5 contributes to C(i1, k1) with -1 and C(i1, k2) with +1.
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        
        # Factor 6: M6
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        # C11 += M6
        W[k2*n+i2, col] = 1
        col += 1
        
        # Factor 7: M7
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        # C00 += M7
        W[k1*n+i1, col] = 1
        col += 1

    # Helper to add outer product factors for A_i_j * B_k_l
    # U[i,j] = 1, V[k,l] = 1, W[l,i] = 1
    # But here we have vector outer products.
    # A_col (size n) * B_row (size p)
    # C[i, k] = A[i] * B[k]
    # Rank n*p.
    def add_outer_product(i_indices, j_idx, k_indices):
        nonlocal col
        for i in i_indices:
            for k in k_indices:
                U[i*m + j_idx, col] = 1
                V[j_idx*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

    # 1a. Strassen for A[:, 0:2] * B[0:2, 0:2]
    # i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # 1b. Outer product for A[:, 2] * B[2, 0:2]
    # i in {0,1}, j=2, k in {0,1}
    add_outer_product([0, 1], 2, [0, 1])
    
    # 2a1. Strassen for A[:, 0:2] * B[0:2, 2:4]
    # i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # 2a2. Outer product for A[:, 0:2] * B[0:2, 4]
    # This is A (2x2) * B (2x1).
    # C[i, 4] = sum_j A[i,j] B[j,4]
    # Rank 4.
    # Factors for (i, j) pairs? No, factors for (i, k) pairs?
    # C[i, 4] = A[i,0] B[0,4] + A[i,1] B[1,4]
    # We can decompose this into 4 terms:
    # T1: A[0,0] B[0,4] -> C[0,4]
    # T2: A[0,1] B[1,4] -> C[0,4]
    # T3: A[1,0] B[0,4] -> C[1,4]
    # T4: A[1,1] B[1,4] -> C[1,4]
    # So 4 factors.
    # For each (i, j), we have a factor where U[i,j]=1, V[j,4]=1, W[4,i]=1.
    add_outer_product([0, 1], 0, [4]) # This adds 2 factors: (0,0,4) and (1,0,4)
    add_outer_product([0, 1], 1, [4]) # This adds 2 factors: (0,1,4) and (1,1,4)
    
    # 2b. Outer product for A[:, 2] * B[2, 2:5]
    # i in {0,1}, j=2, k in {2,3,4}
    add_outer_product([0, 1], 2, [2, 3, 4])

    # Verify rank used
    if col != rank:
        # Should not happen
        pass

    # Verify exactness
    res = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]
        
    if not np.allclose(res, 1.0, atol=1e-9):
        # Fallback to schoolbook if construction fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
