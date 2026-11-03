import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    # Initialize matrices
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each term is a list of (local_i, local_j, coeff) for U
    # (local_j, local_k, coeff) for V
    # (local_k, local_i, coeff) for W
    strassen_terms = [
        # Term 1
        [
            [(0, 0, 1), (1, 1, 1)], # U
            [(0, 0, 1), (1, 1, 1)], # V
            [(0, 0, 1), (1, 1, 1)]  # W
        ],
        # Term 2
        [
            [(1, 0, 1), (1, 1, 1)], # U
            [(0, 0, 1)],             # V
            [(0, 1, 1), (1, 1, -1)]  # W
        ],
        # Term 3
        [
            [(0, 0, 1)],             # U
            [(0, 1, 1), (1, 1, -1)], # V
            [(1, 0, 1), (1, 1, 1)]   # W
        ],
        # Term 4
        [
            [(1, 1, 1)],             # U
            [(1, 0, 1), (0, 0, -1)], # V
            [(0, 0, 1), (0, 1, 1)]   # W
        ],
        # Term 5
        [
            [(0, 0, 1), (0, 1, 1)], # U
            [(1, 1, 1)],             # V
            [(0, 0, -1), (1, 0, 1)]  # W
        ],
        # Term 6
        [
            [(1, 0, 1), (0, 0, -1)], # U
            [(0, 0, 1), (0, 1, 1)],  # V
            [(1, 1, 1)]              # W
        ],
        # Term 7
        [
            [(0, 1, 1), (1, 1, -1)], # U
            [(1, 0, 1), (1, 1, 1)],  # V
            [(0, 0, 1)]              # W
        ]
    ]
    
    col = 0
    # Sub-problem definitions: (i_offset, j_offset, k_offset)
    # i is 0 or 1, j is 0..3, k is 0..3
    # Local indices are 0..1
    subs = [
        (0, 0, 0), # A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        (0, 0, 2), # A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        (0, 2, 0), # A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        (0, 2, 2)  # A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    ]
    
    for i_off, j_off, k_off in subs:
        for term in strassen_terms:
            u_part, v_part, w_part = term
            
            for li, lj, cu in u_part:
                gi = i_off + li
                gj = j_off + lj
                idx_u = gi * m + gj
                U[idx_u][col] += cu
            
            for lj, lk, cv in v_part:
                gj = j_off + lj
                gk = k_off + lk
                idx_v = gj * p + gk
                V[idx_v][col] += cv
            
            for lk, li, cw in w_part:
                gk = k_off + lk
                gi = i_off + li
                idx_w = gk * n + gi
                W[idx_w][col] += cw
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
