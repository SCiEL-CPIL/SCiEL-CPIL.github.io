import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # u_l: [a00, a01, a10, a11]
    # v_l: [b00, b01, b10, b11]
    # w_l: {c00: coeff, c01: coeff, c10: coeff, c11: coeff}
    
    strassen_terms = [
        ( [1, 0, 0, 1], [1, 0, 0, 1], {0: 1, 3: 1} ),
        ( [0, 0, 1, 1], [1, 0, 0, 0], {2: 1, 3: -1} ),
        ( [1, 0, 0, 0], [0, 0, 1, -1], {1: 1, 3: 1} ),
        ( [0, 0, 0, 1], [-1, 1, 0, 0], {0: 1, 2: 1} ),
        ( [1, 1, 0, 0], [0, 0, 0, 1], {0: -1, 1: 1} ),
        ( [-1, 0, 1, 0], [1, 1, 0, 0], {3: 1} ),
        ( [0, 1, 0, -1], [0, 0, 1, 1], {0: 1} )
    ]
    
    # Pre-define indices
    a_idx_0 = [0, 1, 2, 3]
    a_idx_1 = [4, 5, 6, 7]
    
    b_idx_0 = [0, 1, 4, 5]
    b_idx_1 = [2, 3, 6, 7]
    b_idx_2 = [8, 9, 12, 13]
    b_idx_3 = [10, 11, 14, 15]
    
    c_idx_0 = [0, 2, 1, 3]
    c_idx_1 = [4, 6, 5, 7]
    
    blocks = [
        (a_idx_0, b_idx_0, c_idx_0),
        (a_idx_0, b_idx_1, c_idx_1),
        (a_idx_1, b_idx_2, c_idx_0),
        (a_idx_1, b_idx_3, c_idx_1),
    ]
    
    col = 0
    for a_idx, b_idx, c_idx in blocks:
        for u_l, v_l, w_l in strassen_terms:
            if time.time() > deadline:
                break
            
            # Update U
            U[a_idx[0], col] += u_l[0]
            U[a_idx[1], col] += u_l[1]
            U[a_idx[2], col] += u_l[2]
            U[a_idx[3], col] += u_l[3]
            
            # Update V
            V[b_idx[0], col] += v_l[0]
            V[b_idx[1], col] += v_l[1]
            V[b_idx[2], col] += v_l[2]
            V[b_idx[3], col] += v_l[3]
            
            # Update W
            for lc, coeff in w_l.items():
                global_c_idx = c_idx[lc]
                W[global_c_idx, col] += coeff
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
