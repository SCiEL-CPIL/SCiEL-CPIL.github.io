import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Build the matrices efficiently
    for l in range(8):
        for s in range(7):
            col = 7 * l + s
            
            U[:, col] = strassen_U[:, s]
            
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            V[b_rows, col] = strassen_V[:, s]
            
            c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
            W[c_rows, col] = strassen_W[:, s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
