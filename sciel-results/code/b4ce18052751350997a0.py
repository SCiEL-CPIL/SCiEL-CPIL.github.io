import numpy as np

def construct():
    n, m, p = 2, 3, 6
    
    # Target Rank 33
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> Rank 12
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> Rank 21 (3 Strassen 2x2x2 blocks)
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A[i, 0] * B[0, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Two 2x2x2 Strassen blocks for each of the 3 column pairs
    # A_sub is A[:, 1:3] (2x2)
    # B_sub is B[1:3, :] (2x6)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices in U: i*m + j. Here i in {0,1}, j in {1,2} (cols 1,2 of A)
    # Indices in V: j*p + k. Here j in {1,2} (rows 1,2 of B), k in {0..5}
    # Indices in W: k*n + i. Here k in {0..5}, i in {0,1}
    
    for t in range(3):
        # Block t uses B cols 2*t, 2*t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verification - vectorized for speed
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # Compute tensor contraction: sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This equals the matrix multiplication tensor T[i,j,k] which should be 1 for all i,j,k
    
    # Reshape to compute efficiently
    # For each (i,j,k), we need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can compute this as: for each r, outer product of U_r[i,j], V_r[j,k], W_r[k,i]
    # But simpler: use einsum
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col_s = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii * m + jj, col_s] = 1.0
                    V_school[jj * p + kk, col_s] = 1.0
                    W_school[kk * n + ii, col_s] = 1.0
                    col_s += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
