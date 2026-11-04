import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    # Rank 24 decomposition:
    # Part 1: A[:, 0:2] (2x2) * B[0:2, :] (2x6) using Strassen rank-7, each mult has rank 3 -> 21
    # Part 2: A[:, 2] (2x1) * B[2, :] (1x6) using schoolbook -> rank 6
    # Total: 21 + 6 = 27... wait let me recalculate
    
    # Actually each Strassen "multiplication" is (1x3) * (3x2) = 1x2, rank 3
    # 7 such multiplications = 21
    # Plus A[:,2] * B[2,:] = (2x1)*(1x6) = 2x6, rank 6
    # Total = 27
    
    # Let me try to do better. What if I use the full 2x3 structure more cleverly?
    # Actually, let me first verify rank 27 works, then try to optimize.
    
    rank = 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 2 first (simpler): A[:, 2] * B[2, :]
    # A[i, 2] * B[2, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 2, col] = 1.0
            V[2 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 1: Strassen for A[:, 0:2] * B[0:2, :]
    # A_sub is 2x2 (rows 0,1 of A; cols 0,1 of A)
    # B_sub is 2x6 (rows 0,1 of B; all 6 cols)
    # We view B as having 3 blocks of 2 cols each
    
    # Strassen for 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M4
    
    # But here B has 6 columns, so we split into 3 blocks of 2 columns each
    # For each block t (t=0,1,2), B_t is 2x2 (rows 0,1; cols 2t, 2t+1)
    # And we get C_t which is 2x2
    
    # Each "multiplication" like (A00+A11)(B00+B11) where A00+A11 is 1x2 and B00+B11 is 2x2
    # gives a 1x2 result, which has rank 2 (outer product of dim 2)
    
    # Wait, I need to be more careful. A00 is a scalar (A[i=0,j=0]), so A00+A11 is scalar.
    # B00 is 1x2 (B[0, 2t:2t+2]), B11 is 1x2 (B[1, 2t:2t+2])
    # So (A00+A11)*(B00+B11) is scalar * (1x2) = 1x2, rank 2
    
    # Hmm, so each Strassen multiplication has rank 2, not 3.
    # 7 multiplications * 3 blocks * rank 2 = 42. That's worse!
    
    # I think I'm confusing myself. Let me restart with a clearer approach.
    
    # The tensor for 2x3x6 matrix multiplication has rank at most 36 (schoolbook).
    # The current solution achieves 33.
    # Let me try a different decomposition.
    
    # Alternative: use the fact that 6 = 2*3 and try to use Kronecker structure.
    # Or try rank 32, 31, etc.
    
    # For now, let me just return the proven rank-33 solution to be safe.
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> Rank 12
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Two 2x2x2 Strassen blocks for each of the 3 column pairs
    for t in range(3):
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

    # Fast vectorized verification
    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :] * V_r[j, k, :], W_r[k, i, :])
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    if max_err > 1e-9:
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
