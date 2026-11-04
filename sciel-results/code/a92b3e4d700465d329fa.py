import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Define the k-blocks of size 2
    # p=6, so 3 blocks
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # Total rank: m * number_of_k_blocks
    rank = m * len(k_blocks)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    c = 0
    for j in range(m):
        for k_start, k_end in k_blocks:
            # This term corresponds to multiplication A[:, j] * B[j, k_start:k_end+1]
            # U: i varies 0..n-1, j is fixed. U[i*m+j, c] = 1
            for i in range(n):
                U[i * m + j, c] = 1
            
            # V: j is fixed, k varies in k_block. V[j*p+k, c] = 1
            for k in range(k_start, k_end + 1):
                V[j * p + k, c] = 1
            
            # W: k varies in k_block, i varies 0..n-1. W[k*n+i, c] = 1
            for k in range(k_start, k_end + 1):
                for i in range(n):
                    W[k * n + i, c] = 1
            
            c += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
