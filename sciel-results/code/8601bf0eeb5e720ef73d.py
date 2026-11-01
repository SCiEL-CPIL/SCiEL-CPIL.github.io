import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of the schoolbook decomposition
    # Rank index r corresponds to (k, i, j) where r = k*(n*m) + i*m + j
    
    r = np.arange(rank)
    
    # Decompose r into k, i, j
    k_vals = (r // (n * m)).astype(int)
    rem = r % (n * m)
    i_vals = (rem // m).astype(int)
    j_vals = (rem % m).astype(int)
    
    # Calculate row indices for U, V, W
    # U: n x m matrix, index i*m + j
    u_indices = i_vals * m + j_vals
    # V: m x p matrix, index j*p + k
    v_indices = j_vals * p + k_vals
    # W: p x n matrix, index k*n + i (trace convention)
    w_indices = k_vals * n + i_vals
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate matrices
    U[u_indices, r] = 1
    V[v_indices, r] = 1
    W[w_indices, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
