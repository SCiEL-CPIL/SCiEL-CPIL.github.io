import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 110  # safety margin before 120s kill

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen terms: (u_indices, v_indices, w_indices)
    # Indices are relative to the 2x2 block
    strassen_terms = [
        # P0 = (A00 + A11) * (B00 + B11)
        ([0, 3], [0, 3], [0, 3, 1, 2]),
        # P1 = (A10 + A11) * B00
        ([2, 3], [0], [2, 3]),
        # P2 = A00 * (B01 - B11)
        ([0], [1, 3], [1, 3]),
        # P3 = A11 * (B10 - B00)
        ([3], [2, 0], [0, 2]),
        # P4 = (A00 + A01) * B11
        ([0, 1], [3], [1, 3]),
        # P5 = (A10 - A00) * (B00 + B01)
        ([2, 0], [0, 1], [0, 2]),
        # P6 = (A01 - A11) * (B10 + B11)
        ([1, 3], [2, 3], [0, 1]),
    ]
    
    # Coefficients for Strassen terms
    # P0: +1, +1, +1, -1, +1, +1
    # P1: +1, +1, +1, +1
    # P2: +1, +1, -1, +1, +1
    # P3: +1, +1, -1, +1, -1
    # P4: +1, +1, +1, +1, +1
    # P5: +1, -1, +1, +1, -1, +1
    # P6: +1, -1, +1, +1, +1, +1
    
    # This manual coefficient tracking is error prone. 
    # The original code used explicit (idx, coeff) pairs.
    # I will stick to the original logic but with numpy arrays for speed.
    
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]

    strassen_terms_full = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]

    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms_full[t]

            for u_idx, coeff in u_coeffs:
                U_list[u_idx][col] += coeff

            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V_list[v_idx][col] += coeff

            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W_list[w_idx][col] += coeff

    schoolbook_terms = [
        ([ (0, 1) ], [ (8, 1) ],  [ (16, 1) ]),
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),
        ([ (2, 1) ], [ (8, 1) ],  [ (17, 1) ]),
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),
    ]

    col_start = 28
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]

        for u_idx, coeff in u_coeffs:
            U_list[u_idx][col] += coeff

        for v_idx, coeff in v_coeffs:
            V_list[v_idx][col] += coeff

        for w_idx, coeff in w_coeffs:
            W_list[w_idx][col] += coeff

    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS
    
    # Directly return the rank 32 solution
    # The rank 31 attempt is removed as it is mathematically impossible 
    # for this direct-sum structure and wastes time.
    U, V, W = _build_rank_32(n, m, p)
    
    return {"U": U, "V": V, "W": W}
