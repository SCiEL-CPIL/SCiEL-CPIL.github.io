"""Rewritten program using an output-column grouped construction for matrix multiplication tensor."""
DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use a decomposition grouped by output column k.
    # For each k, we compute the k-th column of C: C[:, k] = A @ B[:, k]
    # This corresponds to a 2x3x1 multiplication for each k.
    # The rank is sum over k of rank(2x3x1) = p * (n*m) = 6 * 6 = 36.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
