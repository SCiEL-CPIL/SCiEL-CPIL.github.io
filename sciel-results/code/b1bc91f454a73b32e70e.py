"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Schoolbook baseline
    rank_school = n * m * p
    U_base = [[0] * rank_school for _ in range(n * m)]
    V_base = [[0] * rank_school for _ in range(m * p)]
    W_base = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = rank_school

    # Try to find a lower rank decomposition using Strassen-like blocks
    # For 4x5x6, we can try to exploit structure.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But for 4x5x6, the exact rank is not known, but it's less than 120.
    # Strassen for 2x2x2 has rank 7 vs 8.
    # We can try to build a decomposition using tensor rank properties.

    # Let's try a simple improvement: if we can find any rank < 120, we use it.
    # For now, let's stick with schoolbook but check if we can do better with known results.
    # Actually, for 4x5x6, the rank is at most 120 (schoolbook).
    # There are known algorithms with lower rank for specific dimensions.
    # Let's try to implement a simple heuristic to reduce rank.

    # One approach: use the fact that matrix multiplication is bilinear.
    # We can try to find a sparse decomposition.

    # For now, return the schoolbook decomposition.
    # If we have time, we could try to optimize further.

    return best_result
# EVOLVE-BLOCK-END
