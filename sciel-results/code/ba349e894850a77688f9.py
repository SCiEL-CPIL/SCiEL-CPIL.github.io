"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

    # Try to find a lower-rank decomposition via random search / structured approaches
    # The schoolbook rank is n*m*p = 72.
    # For 3x4x6, we know the minimum rank is at most 72.
    # Let's try to find if there's a known lower bound or use a structured search.
    
    # Known results: For small matrices, the minimum rank can sometimes be less than n*m*p.
    # However, for 3x4x6, let's first try the schoolbook and see if we can do better.
    
    # Actually, let's implement a simple approach:
    # 1. Start with schoolbook (rank 72)
    # 2. Try to verify if a lower rank is possible using random linear algebra
    
    # For now, return the schoolbook decomposition as a valid baseline.
    # The schoolbook decomposition is always valid.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as expected by the interface
    U_list = [[int(round(val)) if abs(val - round(val)) < 1e-9 else round(val * 2) / 2 for val in row] for row in U]
    V_list = [[int(round(val)) if abs(val - round(val)) < 1e-9 else round(val * 2) / 2 for val in row] for row in V]
    W_list = [[int(round(val)) if abs(val - round(val)) < 1e-9 else round(val * 2) / 2 for val in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
