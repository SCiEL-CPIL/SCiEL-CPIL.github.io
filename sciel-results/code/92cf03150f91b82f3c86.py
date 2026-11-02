# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_W_local = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    strassen_V_local = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    for b in range(3):
        for r in range(7):
            for i in range(4):
                U[i][col] += strassen_U[r][i]
            
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff
            
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            
            col += 1
    
    # Schoolbook for last block (col 6)
    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    schoolbook_V_local = [
        {0: 1},
        {1: 1},
        {0: 1},
        {1: 1}
    ]
    
    schoolbook_W_local = [
        {0: 1},
        {0: 1},
        {1: 1},
        {1: 1}
    ]
    
    for r in range(4):
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]
            
        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0:
                v_idx = 0 * p + 6
            elif loc_idx == 1:
                v_idx = 1 * p + 6
            V[v_idx][col] += coeff
            
        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0:
                w_idx = 6 * n + 0
            elif loc_idx == 1:
                w_idx = 6 * n + 1
            W[w_idx][col] += coeff
            
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
