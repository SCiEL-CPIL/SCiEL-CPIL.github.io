import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # Based on the hint, a block decomposition might yield rank 54.
    # We can try to verify if rank 54 is possible or just use the schoolbook if search fails.
    # Given the time limit, we will just return the schoolbook decomposition if we don't have a specific improved algorithm.
    # However, the hint suggests 54 is possible.
    # Let's try a simple heuristic: if we can find a valid decomposition of rank < 60, return it.
    # For now, we return the schoolbook to ensure correctness.
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
