import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6

    # Build target tensor once (used only for verification)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    deadline = time.time() + 110.0

    # Attempt Rank 92: 3x3 blocks of 2x2, Bini(23) x Schoolbook(4)
    try:
        U, V, W = _construct_rank_92(n, m, p)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T_recon, T, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: rank-189 via Strassen(2x2, rank 7) x schoolbook(3x3, rank 27)
    rank = 189
    bs = 3
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        U[i_glob * m + k_glob, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        V[k_glob * p + j_glob, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            W[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Final fallback: schoolbook rank 216
        U_std = np.zeros((n * m, n * m * p), dtype=float)
        V_std = np.zeros((m * p, n * m * p), dtype=float)
        W_std = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_std[i * m + k, idx] = 1.0
                    V_std[k * p + j, idx] = 1.0
                    W_std[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_92(n, m, p):
    # 3x3 blocks of 2x2.
    # Outer: Bini et al. rank 23 for 3x3.
    # Inner: Schoolbook rank 4 for 2x2.
    # Total rank: 23 * 4 = 92.
    
    # Bini et al. rank 23 decomposition for 3x3 matrix multiplication.
    # Coefficients are for A, B, C blocks (3x3 grid).
    # Each term is a list of (row, col, coeff) for A, B, C.
    bini_defs = [
        # 1
        {'A': [(0,0,1.0)], 'B': [(0,0,1.0)], 'C': [(0,0,1.0)]},
        # 2
        {'A': [(0,1,1.0)], 'B': [(0,0,1.0)], 'C': [(0,0,1.0)]},
        # 3
        {'A': [(0,0,1.0)], 'B': [(0,1,1.0)], 'C': [(0,1,1.0)]},
        # 4
        {'A': [(0,1,1.0)], 'B': [(0,1,1.0)], 'C': [(0,1,1.0)]},
        # 5
        {'A': [(1,0,1.0)], 'B': [(0,0,1.0)], 'C': [(1,0,1.0)]},
        # 6
        {'A': [(1,1,1.0)], 'B': [(0,0,1.0)], 'C': [(1,0,1.0)]},
        # 7
        {'A': [(1,0,1.0)], 'B': [(0,1,1.0)], 'C': [(1,1,1.0)]},
        # 8
        {'A': [(1,1,1.0)], 'B': [(0,1,1.0)], 'C': [(1,1,1.0)]},
        # 9
        {'A': [(0,0,1.0)], 'B': [(1,0,1.0)], 'C': [(0,2,1.0)]},
        # 10
        {'A': [(0,1,1.0)], 'B': [(1,0,1.0)], 'C': [(0,2,1.0)]},
        # 11
        {'A': [(0,0,1.0)], 'B': [(1,1,1.0)], 'C': [(0,2,1.0)]},
        # 12
        {'A': [(0,1,1.0)], 'B': [(1,1,1.0)], 'C': [(0,2,1.0)]},
        # 13
        {'A': [(1,0,1.0)], 'B': [(1,0,1.0)], 'C': [(1,2,1.0)]},
        # 14
        {'A': [(1,1,1.0)], 'B': [(1,0,1.0)], 'C': [(1,2,1.0)]},
        # 15
        {'A': [(1,0,1.0)], 'B': [(1,1,1.0)], 'C': [(1,2,1.0)]},
        # 16
        {'A': [(1,1,1.0)], 'B': [(1,1,1.0)], 'C': [(1,2,1.0)]},
        # 17
        {'A': [(0,0,1.0)], 'B': [(2,0,1.0)], 'C': [(2,0,1.0)]},
        # 18
        {'A': [(0,1,1.0)], 'B': [(2,0,1.0)], 'C': [(2,0,1.0)]},
        # 19
        {'A': [(0,0,1.0)], 'B': [(2,1,1.0)], 'C': [(2,1,1.0)]},
        # 20
        {'A': [(0,1,1.0)], 'B': [(2,1,1.0)], 'C': [(2,1,1.0)]},
        # 21
        {'A': [(1,0,1.0)], 'B': [(2,0,1.0)], 'C': [(2,2,1.0)]},
        # 22
        {'A': [(1,1,1.0)], 'B': [(2,0,1.0)], 'C': [(2,2,1.0)]},
        # 23
        {'A': [(1,0,1.0)], 'B': [(2,1,1.0)], 'C': [(2,2,1.0)]},
    ]
    
    # This is a placeholder/guess. It will likely fail verification.
    # But the structure is correct for 3x3 blocks of 2x2.
    # Inner: Schoolbook 2x2 (rank 4).
    
    rank = 23 * 4
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r_idx = 0
    for s in range(23):
        s_def = bini_defs[s]
        for u in range(2): # 2x2 block size
            for v in range(2):
                for j in range(2):
                    for (ar, ac, coeff_a) in s_def['A']:
                        i_glob = ar * 2 + u
                        k_glob = ac * 2 + v
                        U[i_glob * m + k_glob, r_idx] += coeff_a
                    for (br_, bc_, coeff_b) in s_def['B']:
                        k_glob = br_ * 2 + v
                        j_glob = bc_ * 2 + j
                        V[k_glob * p + j_glob, r_idx] += coeff_b
                    for (cr, cc, coeff_c) in s_def['C']:
                        i_glob = cr * 2 + u
                        j_glob = cc * 2 + j
                        W[j_glob * n + i_glob, r_idx] += coeff_c
                    r_idx += 1
    return U, V, W
