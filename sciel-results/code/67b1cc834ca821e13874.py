import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Vectorized construction of the trivial decomposition
    # This creates the full set of basis tensors e_i ⊗ e_j ⊗ e_k
    # which are linearly independent, so rank = n*m*p
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Map (i, j) to row index in U
    u_rows = i_idx * m + j_idx
    # Map (j, k) to row index in V
    v_rows = j_idx * p + k_idx
    # Map (k, i) to row index in W
    w_rows = k_idx * n + i_idx
    
    # Set 1s at the appropriate positions
    # Since each column in U, V, W corresponds to a unique (i,j,k), 
    # we can just use advanced indexing to set 1s.
    # Note: Each column has exactly one 1 in U, one in V, one in W.
    
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
