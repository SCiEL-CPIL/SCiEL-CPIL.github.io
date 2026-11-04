import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = n * m * p  # 72

    # Generate indices for the schoolbook decomposition
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Each rank term corresponds to a triple (i, j, k)
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), 
        np.arange(m), 
        np.arange(p), 
        indexing='ij'
    )
    
    # Flatten the indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Construct U, V, W as sparse-like dense matrices
    # We can use scatter or direct assignment
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U indices: row = i*m + j, col = col_idx
    U_rows = i_flat * m + j_flat
    V_rows = j_flat * p + k_flat
    W_rows = k_flat * n + i_flat
    
    cols = np.arange(rank)
    
    U[U_rows, cols] = 1
    V[V_rows, cols] = 1
    W[W_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
