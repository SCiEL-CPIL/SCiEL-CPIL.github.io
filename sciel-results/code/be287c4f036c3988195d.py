import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, R):
        U_r = U.reshape((n, m, R))
        V_r = V.reshape((m, p, R))
        W_r = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)

    # Attempt Rank 23 (Bini et al.)
    R = 23
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Bini 23 Terms
    # 0: A00 B00 -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # 1: A11 B11 -> C11
    U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
    # 2: A22 B22 -> C22
    U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
    
    # 3: (A01+A11)(B00+B10) -> C00+C01+C10+C11
    U[1, 3] = 1; U[4, 3] = 1
    V[0, 3] = 1; V[3, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1; W[3, 3] = 1; W[4, 3] = 1
    
    # 4: (A02+A12)(B00+B10) -> C00+C02+C10+C12
    U[2, 4] = 1; U[5, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1; W[3, 4] = 1; W[5, 4] = 1
    
    # 5: (A01+A21)(B00+B20) -> C00+C01+C20+C21
    U[1, 5] = 1; U[6, 5] = 1
    V[0, 5] = 1; V[6, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1; W[6, 5] = 1; W[7, 5] = 1
    
    # 6: (A02+A22)(B00+B20) -> C00+C02+C20+C22
    U[2, 6] = 1; U[8, 6] = 1
    V[0, 6] = 1; V[6, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1; W[6, 6] = 1; W[8, 6] = 1
    
    # 7: (A11+A21)(B10+B20) -> C10+C11+C20+C21
    U[4, 7] = 1; U[6, 7] = 1
    V[3, 7] = 1; V[6, 7] = 1
    W[3, 7] = 1; W[4, 7] = 1; W[6, 7] = 1; W[7, 7] = 1
    
    # 8: (A12+A22)(B10+B20) -> C10+C12+C20+C22
    U[5, 8] = 1; U[8, 8] = 1
    V[3, 8] = 1; V[6, 8] = 1
    W[3, 8] = 1; W[5, 8] = 1; W[6, 8] = 1; W[8, 8] = 1
    
    # 9: (A01+A02)(B10+B11) -> C01+C02+C11+C12
    U[1, 9] = 1; U[2, 9] = 1
    V[3, 9] = 1; V[4, 9] = 1
    W[1, 9] = 1; W[2, 9] = 1; W[4, 9] = 1; W[5, 9] = 1
    
    # 10: (A11+A12)(B20+B21) -> C11+C12+C21+C22
    U[4, 10] = 1; U[5, 10] = 1
    V[6, 10] = 1; V[7, 10] = 1
    W[4, 10] = 1; W[5, 10] = 1; W[7, 10] = 1; W[8, 10] = 1
    
    # 11: (A21+A22)(B10+B11) -> C01+C02+C21+C22 (Correction: This term usually targets different C indices)
    # Actually, the Bini 23 set is specific. The above 12 terms are a guess.
    # If verification fails, we fall back.
    
    # Fill remaining 11 terms with zeros to allow verification
    # If this specific set is wrong, verification will fail.
    
    if verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 26 (Strassen + Border)
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Strassen 2x2 core (7 terms)
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1

    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1

    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1

    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1

    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1

    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1

    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1

    # Standard border terms (19 terms)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
    
    if verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return schoolbook()
