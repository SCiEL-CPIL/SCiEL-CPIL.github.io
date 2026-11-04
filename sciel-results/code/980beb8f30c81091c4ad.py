"""Matrix multiplication 5x5x5 with reduced rank decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the 4x4x4 Strassen decomposition
    # Strassen's 7-term algorithm for 2x2x2, recursively applied
    # For 4x4x4, we can use the 49-term decomposition from Strassen's method
    # Actually, let me use a known 4x4x4 decomposition with rank 49
    
    # The standard Strassen-based 4x4x4 decomposition uses rank 49
    # Let me construct it using the tensor decomposition approach
    
    # For now, use the schoolbook 5x5x5 = 125 as fallback
    # But let's try to construct a 78-rank decomposition
    
    # Approach: Use 4x4 Strassen (rank 49) + boundary terms
    # But the boundary terms for 5x5 are complex
    
    # Let me just return the schoolbook for now and try to improve
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists for return
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
