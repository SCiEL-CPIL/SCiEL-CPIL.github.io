import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for each term r
    # r ranges from 0 to rank-1
    r_indices = np.arange(rank)
    
    # Decompose r into (i, j, k)
    # r = i * m * p + j * p + k
    i_r = r_indices // (m * p)
    j_r = (r_indices // p) % m
    k_r = r_indices % p
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Populate U
    # U[i*m + j, r] = 1 if i == i_r[r] and j == j_r[r]
    # We can do this by iterating over r? No, that's a loop.
    # We can use advanced indexing.
    # For each r, the row index in U is i_r[r] * m + j_r[r].
    # We want to set U[row_idx, col_idx] = 1.
    # row_idx = i_r * m + j_r
    # col_idx = r_indices
    U[i_r * m + j_r, r_indices] = 1
    
    # Populate V
    # V[j*p + k, r] = 1 if j == j_r[r] and k == k_r[r]
    # row_idx = j_r * p + k_r
    # col_idx = r_indices
    V[j_r * p + k_r, r_indices] = 1
    
    # Populate W
    # W[k*n + i, r] = 1 if k == k_r[r] and i == i_r[r]
    # row_idx = k_r * n + i_r
    # col_idx = r_indices
    W[k_r * n + i_r, r_indices] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
