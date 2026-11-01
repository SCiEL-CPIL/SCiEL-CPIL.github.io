def verify(U, V, W, tol=1e-9):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                target = 1.0 if i == k else 0.0
                if abs(val - target) > tol:
                    return False
    return True
