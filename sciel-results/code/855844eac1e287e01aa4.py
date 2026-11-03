"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank by removing linearly dependent terms
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor T as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        for a in range(n * m):
            for b in range(m * p):
                for d in range(p * n):
                    T[a, b, d] += U_np[a, c] * V_np[b, c] * W_np[d, c]
    
    # Flatten T to a 2D matrix for rank estimation
    # T as (n*m) x (m*p * p*n)
    T_flat = T.reshape(n * m, m * p * p * n)
    
    # Try to find a lower-rank decomposition using SVD
    # But we need integer/half-integer entries, so this is heuristic
    
    # Simple rank reduction: try to remove terms that are linear combinations
    # This is a placeholder; a proper algorithm would be more complex
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
