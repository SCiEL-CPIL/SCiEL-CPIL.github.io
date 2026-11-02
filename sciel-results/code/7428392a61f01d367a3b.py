"""Matrix multiplication 3x3x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 15
    P = p * n  # 15
    
    rank_schoolbook = n * m * p
    # Initialize with zeros
    U_sb = np.zeros((N, rank_schoolbook))
    V_sb = np.zeros((M, rank_schoolbook))
    W_sb = np.zeros((P, rank_schoolbook))
    
    # Vectorized construction of the schoolbook decomposition
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    # Create index arrays for i, j, k
    # We need to iterate over all combinations of i, j, k to fill the columns
    # Instead of a python loop, we can generate the indices and assign
    # However, since we need to map (i,j,k) to a specific column 'col',
    # and set 1s at specific rows, we can use advanced indexing.
    
    # Generate all combinations of i, j, k
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten these index arrays to correspond to columns
    # The order of columns in the seed was:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       col = i * m * p + j * p + k
    # Let's verify this order.
    # Seed:
    # col = 0
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       ...
    #       col += 1
    # So col = i * (m * p) + j * p + k.
    
    cols = i_idx.ravel() * (m * p) + j_idx.ravel() * p + k_idx.ravel()
    
    # Rows for U: i * m + j
    u_rows = i_idx.ravel() * m + j_idx.ravel()
    # Rows for V: j * p + k
    v_rows = j_idx.ravel() * p + k_idx.ravel()
    # Rows for W: k * n + i
    w_rows = k_idx.ravel() * n + i_idx.ravel()
    
    # Assign 1s
    U_sb[u_rows, cols] = 1
    V_sb[v_rows, cols] = 1
    W_sb[w_rows, cols] = 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
