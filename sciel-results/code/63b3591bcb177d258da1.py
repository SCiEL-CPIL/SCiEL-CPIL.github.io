# EVOLVE-BLOCK-START
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Returns a rank-21 decomposition for M_{2,2,6} using Strassen's algorithm
    on three 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Strassen terms for 2x2 matrix multiplication
    # U[i, r] coefficients for A
    # V[j, r] coefficients for B
    # W[k, r] coefficients for C
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3 (within a block)
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (within a block)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U keys: 0,1,2,3 for A00, A01, A10, A11
        # V keys: 0,1,2,3 for B00, B01, B10, B11 (local)
        # W keys: (i, k) for C_ik (local)
        
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    U = [[0] * 21 for _ in range(4)]
    V = [[0] * 21 for _ in range(12)]
    W = [[0] * 21 for _ in range(12)]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            for idx, val in u_coeffs.items():
                U[idx][r] = val
                
            for l, val in v_coeffs.items():
                # Map local B index to global V index
                # V index = row * p + col
                # l=0 -> B00 -> row 0, col 2b
                # l=1 -> B01 -> row 0, col 2b+1
                # l=2 -> B10 -> row 1, col 2b
                # l=3 -> B11 -> row 1, col 2b+1
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
                
            for (i, k_local), val in w_coeffs.items():
                # Map local C index to global W index
                # W index = col * n + row
                # k_global = 2b + k_local
                # i is row
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
