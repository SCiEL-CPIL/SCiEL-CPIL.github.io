import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2 multiplication
    # u: indices 0,1,2,3 for A00, A01, A10, A11
    # v: local indices 0,1,2,3 for B00, B01, B10, B11
    # w: local indices 0,1,2,3 for C00, C01, C10, C11
    
    terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col = 0
    for b in range(num_blocks):
        # Global offsets for V
        # B00 -> 2*b
        # B01 -> 2*b+1
        # B10 -> 14+2*b
        # B11 -> 14+2*b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # Global offsets for W
        # C00 -> 2*(2*b) + 0 = 4*b
        # C01 -> 2*(2*b+1) + 0 = 4*b+2
        # C10 -> 2*(2*b) + 1 = 4*b+1
        # C11 -> 2*(2*b+1) + 1 = 4*b+3
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx, col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx], col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx], col] += val
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
