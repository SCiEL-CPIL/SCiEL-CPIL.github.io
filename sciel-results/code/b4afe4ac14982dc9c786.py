import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val
            for local_b, val in v_coeffs.items():
                if local_b == 0: global_b = 2 * t
                elif local_b == 1: global_b = 2 * t + 1
                elif local_b == 2: global_b = m * p // 2 + 2 * t
                else: global_b = m * p // 2 + 2 * t + 1
                V[global_b, col] = val
            for local_c, val in w_coeffs.items():
                if local_c == 0: global_w = (2 * t) * n + 0
                elif local_c == 1: global_w = (2 * t + 1) * n + 0
                elif local_c == 2: global_w = (2 * t) * n + 1
                else: global_w = (2 * t + 1) * n + 1
                W[global_w, col] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
