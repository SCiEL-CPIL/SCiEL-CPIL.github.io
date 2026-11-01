import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a dict of {U_indices: coeffs, V_indices: coeffs, W_indices: coeffs}
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices for block b: 
    #   B00 -> 2b
    #   B01 -> 2b+1
    #   B10 -> 12+2b
    #   B11 -> 12+2b+1
    # W indices for block b:
    #   C00 -> 4b
    #   C01 -> 4b+1
    #   C10 -> 4b+2
    #   C11 -> 4b+3
    
    # We define the structure for one block
    # Then we apply it to each of the 6 blocks
    
    strassen_terms = [
        {
            'U': {0: 1, 3: 1},
            'V': {0: 1, 3: 1}, # Local V indices: 0:B00, 1:B01, 2:B10, 3:B11
            'W': {0: 1, 3: 1}  # Local W indices: 0:C00, 1:C01, 2:C10, 3:C11
        },
        {
            'U': {2: 1, 3: 1},
            'V': {0: 1},
            'W': {2: 1, 3: -1}
        },
        {
            'U': {0: 1},
            'V': {1: 1, 3: -1},
            'W': {1: 1, 3: 1}
        },
        {
            'U': {3: 1},
            'V': {0: 1, 1: -1},
            'W': {0: 1, 2: 1}
        },
        {
            'U': {0: 1, 1: 1},
            'V': {3: 1},
            'W': {0: -1, 1: 1}
        },
        {
            'U': {2: 1, 0: -1},
            'V': {0: 1, 1: 1},
            'W': {3: 1}
        },
        {
            'U': {1: 1, 3: -1},
            'V': {2: 1, 3: 1},
            'W': {0: 1, 3: 1}
        }
    ]
    
    for b in range(6):
        for t in range(7):
            col = b * 7 + t
            term = strassen_terms[t]
            
            # U
            for u_idx, coeff in term['U'].items():
                U[u_idx][col] += coeff
                
            # V
            for v_idx, coeff in term['V'].items():
                # Map local v_idx to global
                # 0 -> 2b, 1 -> 2b+1, 2 -> 12+2b, 3 -> 12+2b+1
                if v_idx == 0:
                    gv_idx = 2 * b
                elif v_idx == 1:
                    gv_idx = 2 * b + 1
                elif v_idx == 2:
                    gv_idx = 12 + 2 * b
                else:
                    gv_idx = 12 + 2 * b + 1
                V[gv_idx][col] += coeff
                
            # W
            for w_idx, coeff in term['W'].items():
                # Map local w_idx to global
                # 0 -> 4b, 1 -> 4b+1, 2 -> 4b+2, 3 -> 4b+3
                gw_idx = 4 * b + w_idx
                W[gw_idx][col] += coeff
                
    return {"U": U, "V": V, "W": W}
