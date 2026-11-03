import numpy as np
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 130.0
    
    n, m, p = 4, 5, 5
    
    # Strassen 2x2 factors
    def get_strassen_2x2_factors():
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2
    
    U2, V2, W2 = get_strassen_2x2_factors()
    
    # Build 4x4x4 Strassen factors
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for s in range(7):
        for r in range(7):
            col = s * 7 + r
            
            for p_idx in range(4):
                coeff = U2[p_idx, s]
                if coeff != 0:
                    bi, bj = p_idx // 2, p_idx % 2
                    for di in range(2):
                        for dj in range(2):
                            i = 2 * bi + di
                            j = 2 * bj + dj
                            U4[i * 4 + j, col] += coeff
            
            for p_idx in range(4):
                coeff = V2[p_idx, s]
                if coeff != 0:
                    bi, bj = p_idx // 2, p_idx % 2
                    for di in range(2):
                        for dj in range(2):
                            i = 2 * bi + di
                            j = 2 * bj + dj
                            V4[i * 4 + j, col] += coeff
            
            for p_idx in range(4):
                coeff = W2[p_idx, s]
                if coeff != 0:
                    bi, bj = p_idx // 2, p_idx % 2
                    for di in range(2):
                        for dj in range(2):
                            i = 2 * bi + di
                            j = 2 * bj + dj
                            W4[j * 4 + i, col] += coeff  # C index: k*n+i = j*4+i
    
    # Build full 4x5x5 factors
    r_total = 85
    U = np.zeros((n * m, r_total))
    V = np.zeros((m * p, r_total))
    W = np.zeros((p * n, r_total))
    
    # Part 1: Strassen 4x4x4
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 0:49] = U4[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
    
    # Part 2: A[i,4] * B[4,k] -> C[k,i] for i in 0..3, k in 0..3
    col = 49
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col] = 1.0
            V[4 * 5 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
            col += 1
    
    # Part 3: A[i,j] * B[j,4] -> C[4,i] for i in 0..3, j in 0..3
    col = 65
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1.0
            V[j * 5 + 4, col] = 1.0
            W[4 * 4 + i, col] = 1.0
            col += 1
    
    # Part 4: A[i,4] * B[4,4] -> C[4,i] for i in 0..3
    col = 81
    for i in range(4):
        U[i * 5 + 4, col] = 1.0
        V[4 * 5 + 4, col] = 1.0
        W[4 * 4 + i, col] = 1.0
        col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    # Fallback
                    r_school = n * m * p
                    U_sb = np.zeros((n * m, r_school))
                    V_sb = np.zeros((m * p, r_school))
                    W_sb = np.zeros((p * n, r_school))
                    for c, (ii, jj, kk) in enumerate([(ii, jj, kk) for ii in range(n) for jj in range(m) for kk in range(p)]):
                        U_sb[ii * m + jj, c] = 1.0
                        V_sb[jj * p + kk, c] = 1.0
                        W_sb[kk * n + ii, c] = 1.0
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
