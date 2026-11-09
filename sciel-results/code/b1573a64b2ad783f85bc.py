import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col_offset = 0
    for t in range(4):
        for s in range(2):
            for p_idx in range(7):
                r = col_offset
                x_coeffs, y_coeffs, z_coeffs = strassen_defs[p_idx]
                
                for x_idx, val in x_coeffs.items():
                    u_idx = [2*s, 2*s+1, 4+2*s, 5+2*s][x_idx]
                    U[u_idx][r] += val
                
                for y_idx, val in y_coeffs.items():
                    v_idx = [16*s + 2*t, 16*s + 2*t+1, 16*s + 8 + 2*t, 16*s + 8 + 2*t+1][y_idx]
                    V[v_idx][r] += val
                
                for z_idx, val in z_coeffs.items():
                    w_idx = [4*t, 4*t+2, 4*t+1, 4*t+3][z_idx]
                    W[w_idx][r] += val
                
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
