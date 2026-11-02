import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START = time.time()
DEADLINE = 130.0


def _elapsed():
    return time.time() - _START


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec = 0.0
                for r in range(rank):
                    dec += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(dec - 1.0) > 1e-8:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    # Rank 33 solution: 3 blocks of 2x3x2 rank 11
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 2x3x2 rank 11 terms
    # A: 2x3, B: 3x2, C: 2x2
    # Strassen 2x2x2 core + 4 outer product terms
    terms_2x3x2 = []
    # Strassen terms for A[:,0:2] * B[0:2,:] -> C
    # M1 = (A00+A11)(B00+B11)
    terms_2x3x2.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    # M2 = (A10+A11)B00
    terms_2x3x2.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    # M3 = A00(B01-B11)
    terms_2x3x2.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    # M4 = A11(B10-B00)
    terms_2x3x2.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    # M5 = (A00+A01)B11
    terms_2x3x2.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (A10-A00)(B00+B01)
    terms_2x3x2.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    # M7 = (A01-A11)(B10+B11)
    terms_2x3x2.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    
    # Outer product terms for A[:,2] * B[2,:] -> C
    # A[:,2] is index 2 (A02) and 5 (A12)
    # B[2,:] is index 4 (B20) and 5 (B21)
    # C indices: 0 (C00), 1 (C01), 2 (C10), 3 (C11)
    # A02 * B20 -> C00
    terms_2x3x2.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    # A02 * B21 -> C01
    terms_2x3x2.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    # A12 * B20 -> C10
    terms_2x3x2.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    # A12 * B21 -> C11
    terms_2x3x2.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})

    for block in range(3):
        for t_idx, term in enumerate(terms_2x3x2):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
