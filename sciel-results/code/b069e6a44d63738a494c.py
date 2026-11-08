import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    V_s = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    W_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1}
    ]
    
    col = 0
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_s[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff
                
                for (j_loc, k_loc), coeff in V_s[s].items():
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff
                
                for (i_loc, k_loc), coeff in W_s[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
