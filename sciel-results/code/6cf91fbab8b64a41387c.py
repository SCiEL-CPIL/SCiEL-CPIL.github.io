"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    current_rank = 0
    
    # Group 1: A1 * Btop
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    # Group 2: A2 * Bbottom
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
