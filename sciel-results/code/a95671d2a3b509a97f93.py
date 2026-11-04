import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We iterate over the rank (columns) and determine which row to set to 1 in each matrix
    # The decomposition is the "trivial" or "schoolbook" one.
    # Each term c corresponds to a triple (i, j, k) such that:
    # i = c // (m * p)
    # j = (c // p) % m
    # k = c % p
    
    # Vectorized approach:
    # Create arrays of indices i, j, k for each column c
    c_indices = np.arange(rank)
    
    # Calculate i, j, k for each c
    # i is the first index of A (row)
    # j is the second index of A (col) / first index of B (row)
    # k is the second index of B (col)
    
    # In the seed:
    # i = 0..n-1
    # j = 0..m-1
    # k = 0..p-1
    # c = i * m * p + j * p + k
    
    i_vals = (c_indices // (m * p)).astype(int)
    j_vals = ((c_indices // p) % m).astype(int)
    k_vals = (c_indices % p).astype(int)
    
    # Indices for U: row = i * m + j
    u_rows = i_vals * m + j_vals
    # Indices for V: row = j * p + k
    v_rows = j_vals * p + k_vals
    # Indices for W: row = k * n + i
    w_rows = k_vals * n + i_vals
    
    # Set the 1s
    # U[u_rows, c_indices] = 1
    # Since c_indices is 0..rank-1, we can just use it as column indices
    U[u_rows, c_indices] = 1.0
    V[v_rows, c_indices] = 1.0
    W[w_rows, c_indices] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
