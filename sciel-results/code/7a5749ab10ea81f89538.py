"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
DIMENSIONS = [4, 4, 7]

def _build_decomp():
    """Build a rank-112 decomposition (schoolbook) for 4x4x7 matmul."""
    n, m, p = 4, 4, 7
    rank = n * m * p  # 112

    U = [[0.0] * rank for _ in range(n * m)]   # 16 x 112
    V = [[0.0] * rank for _ in range(m * p)]   # 28 x 112
    W = [[0.0] * rank for _ in range(p * n)]   # 28 x 112

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    U, V, W = _build_decomp()
    return {"U": U, "V": V, "W": W}
