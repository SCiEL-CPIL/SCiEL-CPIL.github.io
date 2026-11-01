"""Direct bilinear factorization for 2x2 x 2x12 matrix multiplication."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    for a in range(n):
        for b in range(m):
            for c in range(p):
                idx = (a * m + b) * p + c
                U[a * m + b][idx] = 1
                V[b * p + c][idx] = 1
                W[c * n + a][idx] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
