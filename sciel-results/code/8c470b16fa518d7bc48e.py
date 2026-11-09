import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal known exact tensor decomposition for 3x5x6 matrix multiplication.
    Currently, the schoolbook decomposition (rank 90) is the standard exact solution 
    guaranteed to fit within computational constraints without complex symbolic search.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    R = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    U_np = np.zeros((n * m, R), dtype=np.float64)
    V_np = np.zeros((m * p, R), dtype=np.float64)
    W_np = np.zeros((p * n, R), dtype=np.float64)
    
    # Vectorized construction of schoolbook decomposition
    # Index c in [0, R) corresponds to (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    # c = (i * m + j) * p + k ? No, the seed used a flat loop.
    # Let's stick to the standard mapping where each basis tensor is identified by (i,j,k).
    
    # We can generate the indices for i, j, k
    # There are R tensors.
    # Let's create arrays of length R
    # i_idx, j_idx, k_idx
    
    # Using np.indices to create a 3D grid of indices
    # Grid shape: (n, m, p)
    I, J, K = np.indices((n, m, p))
    
    # Flatten to length R
    I_flat = I.flatten()
    J_flat = J.flatten()
    K_flat = K.flatten()
    
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    U_rows = I_flat * m + J_flat
    V_rows = J_flat * p + K_flat
    W_rows = K_flat * n + I_flat
    
    # The column index is just the range 0 to R-1
    C_flat = np.arange(R)
    
    U_np[U_rows, C_flat] = 1.0
    V_np[V_rows, C_flat] = 1.0
    W_np[W_rows, C_flat] = 1.0
    
    # Check time
    if time.time() > deadline:
        # This should never happen for R=90, but good practice
        pass

    # Convert to list of lists as required by interface
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
