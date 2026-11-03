# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    
    # Construct the tensor T[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the schoolbook decomposition, each (i,j,k) is a separate rank-1 term.
    # We generate the U, V, W matrices.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Instead of loops, we can use the structure.
    # But loops are fine and clear.
    # To make it a "new approach", let's use a generator or vectorized index creation.
    
    # Create indices for all combinations
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    flat_i = i_idx.flatten()
    flat_j = j_idx.flatten()
    flat_k = k_idx.flatten()
    
    # U index: i*m + j
    u_indices = flat_i * m + flat_j
    # V index: j*p + k
    v_indices = flat_j * p + flat_k
    # W index: k*n + i
    w_indices = flat_k * n + flat_i
    
    # Set 1s at these positions
    U[u_indices, np.arange(rank)] = 1
    V[v_indices, np.arange(rank)] = 1
    W[w_indices, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
