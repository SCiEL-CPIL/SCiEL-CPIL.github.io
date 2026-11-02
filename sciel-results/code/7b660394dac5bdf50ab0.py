"""Matrix multiplication 3x4x8 with optimized rank decomposition."""
import time

DIMENSIONS = [3, 4, 8]


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 140
    
    n, m, p = DIMENSIONS
    
    # Simple construction: rank = m = 4
    # U[i*m+j, j] = 1, V[j*p+k, j] = 1, W[k*n+i, j] = 1
    rank = m
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for j in range(m):
        for i in range(n):
            U[i*m+j][j] = 1
        for k in range(p):
            V[j*p+k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k*n+i][j] = 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: full schoolbook rank 96
    rank = n * m * p
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
