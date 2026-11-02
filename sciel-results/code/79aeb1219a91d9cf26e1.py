"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank for 2x2x5 is 20.
    # Attempt to find a lower-rank decomposition via cyclic/sliding window structure.
    # For now, use rank 20 with a clean implementation.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
