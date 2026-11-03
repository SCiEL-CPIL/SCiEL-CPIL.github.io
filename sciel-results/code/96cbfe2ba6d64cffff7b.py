import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank is 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A and B parts
    A_forms = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    B_forms = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    C_coeffs = [
        [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0]
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
    ]
    
    for b in range(6):
        if b < 3:
            j_offset_A = 0
            j_offset_B = 0
        else:
            j_offset_A = 2
            j_offset_B = 2
            
        k_offset = (b % 3) * 2
        
        for s in range(7):
            col = b * 7 + s
            
            for (i_loc, j_loc), coeff in A_forms[s].items():
                i_global = i_loc
                j_global = j_loc + j_offset_A
                U[i_global * m + j_global][col] += coeff
                
            for (j_loc, k_loc), coeff in B_forms[s].items():
                j_global = j_loc + j_offset_B
                k_global = k_loc + k_offset
                V[j_global * p + k_global][col] += coeff
                
            for i_loc in range(2):
                for k_loc in range(2):
                    coeff = C_coeffs[i_loc][k_loc][s]
                    if coeff != 0:
                        i_global = i_loc
                        k_global = k_loc + k_offset
                        W[k_global * n + i_global][col] += coeff
                        
    if time.time() > deadline:
        pass # The computation is already done
        
    return {"U": U, "V": V, "W": W}
