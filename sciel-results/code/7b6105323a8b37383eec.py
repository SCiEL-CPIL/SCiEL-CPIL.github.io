"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (32, 56)
    
    # Strassen 7-term decomposition for 2x2x2
    # U rows: 0:a00, 1:a01, 2:a10, 3:a11
    # V rows for block t (cols 2t, 2t+1):
    #   2t: b00, 2t+1: b01
    #   16+2t: b10, 16+2t+1: b11
    # W rows for block t (cols 2t, 2t+1):
    #   4t: c00, 4t+1: c10
    #   4t+2: c01, 4t+3: c11
    
    # Strassen terms:
    # M1 = a00 * (b00 - b10)
    # M2 = (a00 + a10) * b10
    # M3 = a01 * (b11 - b01)
    # M4 = a11 * (b00 + b01)
    # M5 = (a01 + a11) * b11
    # M6 = (a10 - a00) * (b00 + b11)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    U_terms = [
        {0: 1},                # M1: a00
        {0: 1, 2: 1},          # M2: a00 + a10
        {1: 1},                # M3: a01
        {3: 1},                # M4: a11
        {1: 1, 3: 1},          # M5: a01 + a11
        {2: 1, 0: -1},         # M6: a10 - a00
        {1: 1, 3: -1}          # M7: a01 - a11
    ]
    
    V_terms = [
        {0: 1, 16: -1},        # M1: b00 - b10
        {16: 1},               # M2: b10
        {17: 1, 1: -1},        # M3: b11 - b01
        {0: 1, 1: 1},          # M4: b00 + b01
        {17: 1},               # M5: b11
        {0: 1, 17: 1},         # M6: b00 + b11
        {16: 1, 17: 1}         # M7: b10 + b11
    ]
    
    W_terms = [
        {0: 1, 3: 1},          # M1: c00 +1, c11 +1
        {1: 1, 3: -1},         # M2: c10 +1, c11 -1
        {2: 1, 3: 1},          # M3: c01 +1, c11 +1
        {0: 1, 1: 1},          # M4: c00 +1, c10 +1
        {0: -1, 2: 1},         # M5: c00 -1, c01 +1
        {3: 1},                # M6: c11 +1
        {0: 1}                 # M7: c00 +1
    ]
    
    for t in range(8):
        for s in range(7):
            col = t * 7 + s
            
            for r, val in U_terms[s].items():
                U[r][col] = val
                
            base_v = 2 * t
            for r, val in V_terms[s].items():
                V[base_v + r][col] = val
                
            base_w = 4 * t
            for r, val in W_terms[s].items():
                W[base_w + r][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
