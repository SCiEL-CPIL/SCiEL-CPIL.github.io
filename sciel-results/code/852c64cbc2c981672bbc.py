"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total = n * m * p
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    idx = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][idx] = 1.0
                V[jj * p + kk][idx] = 1.0
                W[kk * n + ii][idx] = 1.0
                idx += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
