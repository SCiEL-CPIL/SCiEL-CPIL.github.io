"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses 4 independent Strassen rank-7 decompositions, total rank 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    strassen_terms = [
        # M1 = (a+d)(e+h)
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (c+d)*e
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1), (1, 1, -1)]
        },
        # M3 = a*(f-h)
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(1, 0, 1), (1, 1, 1)]
        },
        # M4 = d*(g-e)
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (0, 1, 1)]
        },
        # M5 = (a+b)*h
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)]
        },
        # M6 = (c-a)*(e+f)
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # M7 = (b-d)*(g+h)
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    blocks = [
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    ]
    
    col = 0
    for j_off, k_off in blocks:
        for term in strassen_terms:
            for (li, lj, coeff) in term['U']:
                global_i = li
                global_j = lj + j_off
                U[global_i * m + global_j][col] += coeff
            
            for (lj, lk, coeff) in term['V']:
                global_j = lj + j_off
                global_k = lk + k_off
                V[global_j * p + global_k][col] += coeff
            
            for (lk, li, coeff) in term['W']:
                global_k = lk + k_off
                global_i = li
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
