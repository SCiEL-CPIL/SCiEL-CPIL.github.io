import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # Each term: (u_dict, v_dict, w_dict) with local indices 0-8
    # u: i*3+j, v: j*3+k, w: k*3+i for i,j,k in {0,1,2}
    laderman_terms = [
        ({0: 1}, {3: 1, 4: -1}, {3: 1, 6: -1}),
        ({0: 1, 1: 1}, {4: 1}, {3: 1, 4: 1}),
        ({4: 1, 5: 1}, {8: 1}, {7: 1, 8: 1}),
        ({8: 1}, {6: 1, 0: -1}, {2: 1, 0: -1}),
        ({6: 1, 8: 1}, {0: 1}, {2: 1, 0: 1}),
        ({2: 1, 8: 1}, {6: 1}, {0: 1, 2: 1}),
        ({0: 1, 4: -1}, {4: 1, 8: 1}, {3: 1, 6: 1, 4: -1, 7: -1}),
        ({4: 1, 8: -1}, {8: 1, 0: 1}, {7: 1, 1: 1, 8: -1, 2: -1}),
        ({8: 1, 0: -1}, {0: 1, 4: 1}, {2: 1, 5: 1, 0: -1, 3: -1}),
        ({1: 1, 4: 1}, {5: 1}, {6: 1, 7: 1}),
        ({4: 1, 7: 1}, {7: 1}, {4: 1, 5: 1}),
        ({7: 1, 1: 1}, {4: 1}, {5: 1, 3: 1}),
        ({2: 1, 5: 1}, {6: 1}, {0: 1, 1: 1}),
        ({5: 1, 8: 1}, {2: 1}, {7: 1, 8: 1}),
        ({8: 1, 2: 1}, {6: 1}, {0: 1, 2: 1}),
        ({0: 1, 3: 1}, {1: 1}, {3: 1, 4: 1}),
        ({3: 1, 6: 1}, {5: 1}, {6: 1, 7: 1}),
        ({6: 1, 0: 1}, {7: 1}, {1: 1, 2: 1}),
        ({1: 1, 3: 1}, {4: 1}, {3: 1, 4: 1}),
        ({4: 1, 6: 1}, {7: 1}, {6: 1, 7: 1}),
        ({7: 1, 0: 1}, {1: 1}, {1: 1, 2: 1}),
        ({2: 1, 3: 1}, {6: 1}, {0: 1, 1: 1}),
        ({5: 1, 6: 1}, {2: 1}, {7: 1, 8: 1}),
    ]
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block_offset in [0, 3]:
        for u_map, v_map, w_map in laderman_terms:
            for u_idx, coeff in u_map.items():
                U[u_idx][col] += coeff
            for v_idx, coeff in v_map.items():
                j = v_idx // 3
                k = v_idx % 3
                V[j * p + (k + block_offset)][col] += coeff
            for w_idx, coeff in w_map.items():
                k = w_idx // 3
                i = w_idx % 3
                W[(k + block_offset) * n + i][col] += coeff
            col += 1
    
    # Verify correctness using numpy for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]))
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U, "V": V, "W": W}
