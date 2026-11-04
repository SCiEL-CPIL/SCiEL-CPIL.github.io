import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-30 tensor decomposition for 2x3 * 3x5 matrix multiplication
    using a Kronecker-product structure.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We construct U, V, W as numpy arrays for efficiency, then convert to lists if needed
    # The interface expects lists of lists or arrays. The seed used lists.
    # Let's stick to lists to be safe, but build them efficiently.
    
    # Index mappings:
    # r -> (i, j, k) where r = i*m*p + j*p + k
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # To satisfy the "Kronecker-product structure" strategy description,
    # we conceptually view this as the sum of outer products of basis vectors.
    # We can generate the indices using vectorized operations.
    
    # Generate all (i, j, k) tuples
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten to get r indices
    r_indices = (i_vals * m * p + j_vals * p + k_vals).flatten()
    
    # Calculate row indices for U, V, W
    u_rows = (i_vals * m + j_vals).flatten()
    v_rows = (j_vals * p + k_vals).flatten()
    w_rows = (k_vals * n + i_vals).flatten()
    
    # Fill the matrices
    # Since we are using Python lists in the current seed, we can do this:
    for idx in range(rank):
        r = r_indices[idx]
        u = u_rows[idx]
        v = v_rows[idx]
        w = w_rows[idx]
        
        U[u][r] = 1
        V[v][r] = 1
        W[w][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
