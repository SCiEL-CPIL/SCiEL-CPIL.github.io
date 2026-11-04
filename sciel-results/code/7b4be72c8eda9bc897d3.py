import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    n, m, p = DIMENSIONS
    rank = n * m * p

    # Pre-allocate matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Schoolbook decomposition: rank = n*m*p
    # Each term r corresponds to a unique (i, j, k)
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    # Vectorized filling for efficiency
    # Generate all (i, j, k) combinations
    # i in [0, n), j in [0, m), k in [0, p)
    # r = (i*m + j)*p + k
    
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Calculate column indices for each factor
    u_cols = i_idx * m + j_idx
    v_cols = j_idx * p + k_idx
    w_cols = k_idx * n + i_idx
    
    # The column index in the factor matrices (which have 'rank' columns) is just the term index r
    # r = (i*m + j)*p + k
    r_idx = (i_idx * m + j_idx) * p + k_idx
    
    # Set values to 1.0
    U[u_cols, r_idx] = 1.0
    V[v_cols, r_idx] = 1.0
    W[w_cols, r_idx] = 1.0

    # Check time (though this is static, we comply with directive)
    if time.time() > deadline:
        # In a real search, we would return best found. Here we just return the static result.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
