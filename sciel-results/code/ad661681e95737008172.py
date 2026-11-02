import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Strassen patterns for 2x2x2
    # Indices are local (i, j) for A, (j, k) for B, (i, k) for C
    # i, j, k in {0, 1}
    
    strassen_terms = [
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]

    # Calculate total rank
    rank = 0
    # Block 1: j in {0,1}, 3 k-blocks of size 2, Strassen rank 7 per k-block
    rank += 3 * 7
    # Block 2: j in {2,3}, 3 k-blocks of size 2, Strassen rank 7 per k-block
    rank += 3 * 7
    # Block 3: j in {4}, schoolbook 2*1*6
    rank += 2 * 1 * 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_block(j_offset, k_offset, term_list):
        nonlocal col
        for term in term_list:
            # U
            for i, j, coeff in term['U']:
                global_i = i
                global_j = j_offset + j
                U[global_i * m + global_j][col] += coeff
            # V
            for j, k, coeff in term['V']:
                global_j = j_offset + j
                global_k = k_offset + k
                V[global_j * p + global_k][col] += coeff
            # W
            for i, k, coeff in term['W']:
                global_i = i
                global_k = k_offset + k
                W[global_k * n + global_i][col] += coeff
            col += 1
            
    # Block 1: j in {0, 1}
    for k_off in [0, 2, 4]:
        add_block(0, k_off, strassen_terms)
        
    # Block 2: j in {2, 3}
    for k_off in [0, 2, 4]:
        add_block(2, k_off, strassen_terms)
        
    # Block 3: j in {4}
    # Schoolbook: A[i,4] * B[4,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            # U[i*m+4] = 1
            U[i * m + 4][col] = 1
            # V[4*p+k] = 1
            V[4 * p + k][col] = 1
            # W[k*n+i] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
