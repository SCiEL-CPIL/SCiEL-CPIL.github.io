import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Approach: Vectorized construction of the trivial decomposition.
    # This uses NumPy broadcasting and array indexing instead of explicit Python loops.
    # The decomposition is the set of all basis tensors $e_i \otimes e_j \otimes e_k$.
    # We construct the matrices U, V, W such that 
    # sum_r U[:,r] V[:,r]^T W[:,r]^T = M_{n,m,p}
    
    # Create index arrays for all combinations of i, j, k
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    rank = len(i_idx)
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Calculate flattened indices
    u_flat = i_idx * m + j_idx
    v_flat = j_idx * p + k_idx
    w_flat = k_idx * n + i_idx
    
    # Set the ones
    U[u_flat, np.arange(rank)] = 1.0
    V[v_flat, np.arange(rank)] = 1.0
    W[w_flat, np.arange(rank)] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
