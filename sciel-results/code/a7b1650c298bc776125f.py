import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # 120 seconds limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Schoolbook decomposition
    # For each i in n, j in m, k in p:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 10:
                    # Safety margin: return current best (which is complete if we finished loops)
                    # If we are in the middle, we might return an incomplete decomposition.
                    # But for rank 50, it finishes in microseconds.
                    pass
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as required by the interface (implied by seed)
    # The seed returned lists. The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't explicitly forbid numpy arrays, but "Keep the function names and signatures ... return interface EXACTLY as in the current program".
    # The current program returns a dict of lists.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
