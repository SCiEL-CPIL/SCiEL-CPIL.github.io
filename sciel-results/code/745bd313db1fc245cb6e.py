# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    
    # Attempt to find a lower rank decomposition (e.g., Rank 23)
    # This is a placeholder for a more complex solver.
    # For now, we return the known Rank 27 decomposition.
    rank = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
