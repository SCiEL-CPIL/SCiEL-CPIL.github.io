for i in range(n):
    for j in range(m):
        for k in range(p):
            s = 0
            for c in range(rank):
                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if abs(s - 1) > 1e-9:
                return False
