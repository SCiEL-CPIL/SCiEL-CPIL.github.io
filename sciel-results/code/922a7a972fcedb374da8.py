"""Program for exact tensor decomposition of matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W by accumulating blocks.
    # Each block is a decomposition of A_sub * B_sub = C_sub.
    
    all_terms = []
    
    # Helper to add a Strassen 2x2x2x2 block
    def add_strassen_block(i1, j1, k1):
        terms = []
        
        # M1 = (a+d)(e+h)
        terms.append((
            {(i1*m + j1): 1, (i1+1)*m + j1+1: 1},
            {(j1*p + k1): 1, (j1+1)*p + k1+1: 1},
            {(k1*n + i1): 1, (k1+1)*n + i1+1: 1}
        ))
        
        # M2 = (c+d)e
        terms.append((
            {(i1+1)*m + j1: 1, (i1+1)*m + j1+1: 1},
            {(j1*p + k1): 1},
            {(k1*n + i1+1): 1}
        ))
        
        # M3 = a(f-h)
        terms.append((
            {(i1*m + j1): 1},
            {(j1*p + k1+1): 1, (j1+1)*p + k1+1: -1},
            {(k1+1)*n + i1: 1}
        ))
        
        # M4 = d(g-e)
        terms.append((
            {(i1+1)*m + j1+1: 1},
            {(j1+1)*p + k1: 1, (j1*p + k1): -1},
            {(k1*n + i1): 1, (k1*n + i1+1): 1}
        ))
        
        # M5 = (a+b)h
        terms.append((
            {(i1*m + j1): 1, (i1*m + j1+1): 1},
            {(j1+1)*p + k1+1: 1},
            {(k1*n + i1): -1, (k1+1)*n + i1: 1}
        ))
        
        # M6 = (c-a)(e+f)
        terms.append((
            {(i1+1)*m + j1: 1, (i1*m + j1): -1},
            {(j1*p + k1): 1, (j1*p + k1+1): 1},
            {(k1+1)*n + i1+1: 1}
        ))
        
        # M7 = (b-d)(g+h)
        terms.append((
            {(i1*m + j1+1): 1, (i1+1)*m + j1+1: -1},
            {(j1+1)*p + k1: 1, (j1+1)*p + k1+1: 1},
            {(k1*n + i1): 1}
        ))
        
        return terms

    # Helper to add simple multiplication block terms
    def add_simple_block_terms(i_start, i_end, j, k_start, k_end):
        terms = []
        for i in range(i_start, i_end):
            for k in range(k_start, k_end):
                terms.append((
                    {(i*m + j): 1},
                    {(j*p + k): 1},
                    {(k*n + i): 1}
                ))
        return terms

    # Block 1: A1 (2x2) * B11 (2x2)
    all_terms.extend(add_strassen_block(0, 0, 0))
    
    # Block 2: A1 (2x2) * B12 (2x2)
    all_terms.extend(add_strassen_block(0, 0, 2))
    
    # Block 3: A1 (2x2) * B13a (2x2)
    all_terms.extend(add_strassen_block(0, 0, 4))
    
    # Block 4: A1 (2x2) * B13b (2x1)
    all_terms.extend(add_simple_block_terms(0, 2, 0, 6, 7))
    all_terms.extend(add_simple_block_terms(0, 2, 1, 6, 7))
    
    # Block 5: A2 (2x1) * B21 (1x2)
    all_terms.extend(add_simple_block_terms(0, 2, 2, 0, 2))
    
    # Block 6: A2 (2x1) * B22 (1x2)
    all_terms.extend(add_simple_block_terms(0, 2, 2, 2, 4))
    
    # Block 7: A2 (2x1) * B23 (1x3)
    all_terms.extend(add_simple_block_terms(0, 2, 2, 4, 7))

    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for row, val in u_dict.items():
            U[row][col] = val
        for row, val in v_dict.items():
            V[row][col] = val
        for row, val in w_dict.items():
            W[row][col] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
