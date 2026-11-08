import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, use 115s to be safe
    
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    products = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, [(0,0,1), (1,1,-1)]),
        ({2: 1, 3: 1}, {0: 1}, [(1,0,1), (1,1,-1)]),
        ({0: 1}, {1: 1, 3: -1}, [(0,1,1)]),
        ({3: 1}, {2: 1, 0: -1}, [(0,0,-1), (1,0,1)]),
        ({0: 1, 1: 1}, {3: 1}, [(1,1,1)]),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, [(0,0,1), (1,1,1)]),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, [(0,0,-1), (0,1,1)])
    ]
    
    col = 0
    for j1 in range(2):
        x_map = {
            0: (0, 2*j1),
            1: (0, 2*j1 + 1),
            2: (1, 2*j1),
            3: (1, 2*j1 + 1)
        }
        
        for scalar_coeffs, vector_coeffs, targets in products:
            for t in range(3):
                if time.time() > deadline:
                    break
                
                for x_idx, coeff in scalar_coeffs.items():
                    i, j = x_map[x_idx]
                    U[i * m + j, col] = coeff
                
                for v_idx, coeff in vector_coeffs.items():
                    row_offset = 2 * j1
                    if v_idx == 0:
                        b_row = row_offset
                        b_col = t
                    elif v_idx == 1:
                        b_row = row_offset
                        b_col = 3 + t
                    elif v_idx == 2:
                        b_row = row_offset + 1
                        b_col = t
                    elif v_idx == 3:
                        b_row = row_offset + 1
                        b_col = 3 + t
                    
                    V[b_row * p + b_col, col] = coeff
                
                for i_z, j2_z, sign in targets:
                    c_row = i_z
                    c_col = j2_z * 3 + t
                    W[c_col * n + c_row, col] = sign
                
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
