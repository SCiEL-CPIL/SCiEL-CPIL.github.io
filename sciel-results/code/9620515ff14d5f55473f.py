import numpy as np

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m x p using a vectorized construction approach.
    
    This implementation uses NumPy to build the rank-60 schoolbook decomposition
    efficiently, distinct from the iterative scalar approach of the seed program.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = 2, 5, 6
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Generate all (i, j, k) indices
    # i in [0, n), j in [0, m), k in [0, p)
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten to get a list of tuples (i, j, k)
    # The order of flattening determines the column index in U, V, W
    # We use C-order (last index changes fastest) to match standard iteration
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate linear indices for U, V, W
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # The column index in the decomposition is the index in the flattened array
    col_idx = np.arange(rank)
    
    # Set the 1s in the factor matrices
    U[u_idx, col_idx] = 1.0
    V[v_idx, col_idx] = 1.0
    W[w_idx, col_idx] = 1.0
    
    # Convert to lists of lists to match the expected output format if required,
    # though NumPy arrays are often accepted. The seed returned lists.
    # To be safe and distinct but compatible, we return lists.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
