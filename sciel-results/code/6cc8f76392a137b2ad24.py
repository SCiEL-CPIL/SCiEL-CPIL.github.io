import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Construct Strassen rank-42 solution
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (0, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for r in range(3):
        for s in range(2):
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]
                for (li, coeff) in u_local:
                    i = li >> 1
                    jl = li & 1
                    jg = 2 * s + jl
                    U[i * m + jg, col] = coeff
                for (lj, coeff) in v_local:
                    jl = lj >> 1
                    kl = lj & 1
                    jg = 2 * s + jl
                    kg = 2 * r + kl
                    V[jg * p + kg, col] = coeff
                for (li, coeff) in w_local:
                    i = li >> 1
                    kl = li & 1
                    kg = 2 * r + kl
                    W[kg * n + i, col] = coeff
                col += 1
    
    # Verify
    T = np.ones((n, m, p))
    T_verify = np.einsum('ijr,jkr,ki->ijk',
                         U.reshape(n, m, rank),
                         V.reshape(m, p, rank),
                         W.reshape(p, n, rank))
    
    if not np.allclose(T_verify, T, atol=1e-9):
        # Fallback
        r_sb = n * m * p
        U_sb = np.zeros((n * m, r_sb))
        V_sb = np.zeros((m * p, r_sb))
        W_sb = np.zeros((p * n, r_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1
                    V_sb[j * p + k, c] = 1
                    W_sb[k * n + i, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

