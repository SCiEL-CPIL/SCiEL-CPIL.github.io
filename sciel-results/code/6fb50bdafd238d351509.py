import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 4, 4, 9
    rank = n * m * p
    
    # Generate indices for the schoolbook decomposition
    # Each term in the sum corresponds to a unique (i, j, k) triple
    # U[i*m + j, r] = 1, V[j*p + k, r] = 1, W[k*n + i, r] = 1
    
    # Create grids for i, j, k
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten to get 1D arrays of indices
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate linear indices for the factor matrices
    # U is (n*m, rank). Row index is i*m + j
    u_rows = i_flat * m + j_flat
    # V is (m*p, rank). Row index is j*p + k
    v_rows = j_flat * p + k_flat
    # W is (p*n, rank). Row index is k*n + i
    w_rows = k_flat * n + i_flat
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Assign 1s to the appropriate positions
    # Each column r corresponds to the r-th (i,j,k) triple
    cols = np.arange(rank)
    U[u_rows, cols] = 1.0
    V[v_rows, cols] = 1.0
    W[w_rows, cols] = 1.0
    
    # Check time limit (though this is fast)
    if time.time() > deadline - 5:
        pass # Return immediately
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
