"""Program for matrix multiplication 3x3 x 3x8 tensor decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rows_u = n * m
    total_rows_v = m * p
    total_rows_w = p * n
    r = n * m * p

    U = [[0.0] * r for _ in range(total_rows_u)]
    V = [[0.0] * r for _ in range(total_rows_v)]
    W = [[0.0] * r for _ in range(total_rows_w)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                U[row_u][idx] = 1
                V[row_v][idx] = 1
                W[row_w][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
