import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 35 decomposition using Strassen for 5 blocks of 2x2
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication
    # A vector indices: 0: A00, 1: A01, 2: A10, 3: A11
    # B vector indices: 0: B00, 1: B01, 2: B10, 3: B11
    # C vector indices: 0: C00, 1: C10, 2: C01, 3: C11
    # Note: W index is k*n + i. 
    # For block k (cols 2k, 2k+1):
    # C[0, 2k] -> W[2k*2 + 0] = W[4k]
    # C[1, 2k] -> W[2k*2 + 1] = W[4k+1]
    # C[0, 2k+1] -> W[(2k+1)*2 + 0] = W[4k+2]
    # C[1, 2k+1] -> W[(2k+1)*2 + 1] = W[4k+3]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    col = 0
    for k in range(5):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A is 2x2, indices [i*m+j]
            for i in range(n * m):
                U[i][col] = a_vec[i]
            
            # V: B is 2x10. Block k uses columns 2k and 2k+1.
            # B[0, 2k] -> V[0*p + 2k]
            # B[0, 2k+1] -> V[0*p + 2k+1]
            # B[1, 2k] -> V[1*p + 2k]
            # B[1, 2k+1] -> V[1*p + 2k+1]
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]
            
            # W: C is 2x10. Block k outputs to columns 2k and 2k+1.
            # C[0, 2k] -> W[2k*n + 0]
            # C[1, 2k] -> W[2k*n + 1]
            # C[0, 2k+1] -> W[(2k+1)*n + 0]
            # C[1, 2k+1] -> W[(2k+1)*n + 1]
            W[2 * k * n + 0][col] = c_vec[0]
            W[2 * k * n + 1][col] = c_vec[1]
            W[(2 * k + 1) * n + 0][col] = c_vec[2]
            W[(2 * k + 1) * n + 1][col] = c_vec[3]
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
