import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    The minimal rank is 49, achieved by applying the rank-7 Strassen decomposition
    to each of the 7 disjoint 2x2 blocks of the 2x14 matrix.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # The rank-7 Strassen terms for 2x2x2 multiplication
    # U, V, W indices are 0..3 for 2x2 matrices (row-major: 0,1,2,3)
    # Terms: (U_dict, V_dict, W_dict)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    num_blocks = p // 2  # 7
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank  # 49
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        k0 = 2 * t
        k1 = k0 + 1
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            u_coeffs, v_local, w_local = strassen_terms[s]
            
            # Fill U
            for u_idx, coeff in u_coeffs.items():
                U[u_idx][col] = coeff
            
            # Fill V
            # v_local keys are 0,1,2,3 corresponding to B[0,0], B[0,1], B[1,0], B[1,1] in the block
            # B is 2x14. Block t corresponds to columns k0, k1.
            # B[j, k0 + v_col] where v_loc = j*2 + v_col
            for v_loc, coeff in v_local.items():
                j = v_loc // 2
                v_col = v_loc % 2
                k_col = k0 + v_col
                v_idx = j * p + k_col
                V[v_idx][col] = coeff
            
            # Fill W
            # w_local keys are 0,1,2,3 corresponding to C[0,0], C[0,1], C[1,0], C[1,1] in the block
            # C is 14x2. Block t corresponds to rows k0, k1.
            # C[k0 + w_row, i] where w_loc = w_row*2 + i
            for w_loc, coeff in w_local.items():
                w_row = w_loc // 2
                i = w_loc % 2
                k_row = k0 + w_row
                w_idx = k_row * n + i
                W[w_idx][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
