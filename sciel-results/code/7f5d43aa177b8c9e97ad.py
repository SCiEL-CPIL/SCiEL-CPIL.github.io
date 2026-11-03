for i in range(n):
    for j in range(m):
        for k in range(p):
            s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
            if abs(s - 1) > tol: return False
