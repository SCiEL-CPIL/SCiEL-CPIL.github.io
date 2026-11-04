"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays for faster construction, then convert to lists
    U_np = np.zeros((n * m, rank), dtype=int)
    V_np = np.zeros((m * p, rank), dtype=int)
    W_np = np.zeros((p * n, rank), dtype=int)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U_np[i * m + j, c] = 1
                # V index: j*p + k
                V_np[j * p + k, c] = 1
                # W index: k*n + i
                W_np[k * n + i, c] = 1
                c += 1
                
    # Convert to lists of lists as required by the interface
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
