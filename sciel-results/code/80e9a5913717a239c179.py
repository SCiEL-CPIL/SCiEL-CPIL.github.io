for i in range(n):
    for j in range(m):
    for k in range(p):
        val = 0.0
        for c in range(rank):
            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(val - 1.0) > tol:
            return False
