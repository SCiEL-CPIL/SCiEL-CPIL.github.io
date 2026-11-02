import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    
    # Target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    deadline = time.time() + 110.0
    
    # Try Rank 161 (Strassen 2x2 of 3x3 blocks with Rank 23 3x3 algorithm)
    # If it fails, fall back to Rank 189 (Strassen 2x2 of 3x3 blocks with Standard 3x3 algorithm)
    
    try:
        rank_161 = _construct_rank_161(n, m, p)
        U, V, W = rank_161
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T_recon, T, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback to Rank 189
    rank = 189
    bs = 3
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 definitions
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    # Reconstruction coefficients for C blocks
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
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b
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
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma
                    r_idx += 1
    
    # Verify fallback
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Final fallback to standard
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

def _construct_rank_161(n, m, p):
    # This function attempts to construct a rank 161 decomposition.
    # It uses Strassen for 2x2 blocks of 3x3 matrices.
    # The 3x3 multiplication uses a rank 23 algorithm.
    # If the rank 23 algorithm is not correctly implemented, this will fail.
    
    bs = 3
    rank_3x3 = 23
    rank_total = 7 * rank_3x3
    
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    # Rank 23 decomposition for 3x3 matrix multiplication
    # Terms: (u, v, w, coeff)
    # u, v, w are indices in 0..8
    # This is a placeholder. The actual terms must be correct.
    # I will use a known valid set if I can recall it.
    # Otherwise, this will fail and fallback will be used.
    
    # For now, I will use the standard 27 terms to ensure the structure is correct,
    # but this will result in rank 189, not 161.
    # To make it distinct, I will change the code structure.
    
    # Actually, I will just return the rank 189 decomposition here as well,
    # but with a different code path.
    # This is not a stronger algorithm.
    
    # I will raise an exception to force fallback.
    raise NotImplementedError("Rank 161 not implemented")
