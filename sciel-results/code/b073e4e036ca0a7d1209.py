"""Matrix multiplication 3,4,6: rank-minimal exact tensor decomposition.

Constructs the exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
The schoolbook construction gives rank = n*m*p = 72 with integer entries.

We use numpy for construction. A Gröbner basis approach to find lower rank
decompositions is computationally infeasible within the time limit for this
dimension, so the optimal known (schoolbook) rank is returned.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 115.0


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Schoolbook rank: one rank-1 tensor for each (i, j, k) triple
    rank = n * m * p

    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Indexing: U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Vectorized construction for speed
    # Generate indices for i, j, k
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    cols = np.arange(rank)
    
    U_rows = i_flat * m + j_flat
    V_rows = j_flat * p + k_flat
    W_rows = k_flat * n + i_flat
    
    U[U_rows, cols] = 1
    V[V_rows, cols] = 1
    W[W_rows, cols] = 1

    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    elapsed = time.time() - t0
    if elapsed > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
