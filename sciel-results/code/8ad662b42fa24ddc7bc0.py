import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-189 decomposition for 6x6 matrix multiplication.
    Uses hierarchical Strassen: 2x2 blocks of size 3x3.
    The inner 3x3 multiplications are decomposed with the
    rank-27 schoolbook algorithm.
    """
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    deadline = time.time() + 140

    # Block size
    bs = 3

    # Rank of inner 3x3 decomposition (schoolbook)
    rank3 = 27

    # Total rank: 7 Strassen products * 27 inner multiplications
    rank = 7 * rank3

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen definitions
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
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a

                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b

                    for c_block_idx in range(4):
                        coeffs = recon_coeffs[c_block_idx]
                        gamma = 0.0
                        for (s_idx, g) in coeffs:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row][r_idx] += gamma

                    r_idx += 1

    assert r_idx == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
