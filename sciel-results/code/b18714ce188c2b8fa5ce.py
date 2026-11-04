def verify(U, V, W, n, m, p):
    r = len(U[0])
    # Check sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c] == delta_{j,j'} for all i,k,j,j'
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(r):
                        s += W[k*n+i][c] * U[i*m+j][c] * V[jp*p+k][c]
                    expected = 1 if j == jp else 0
                    if abs(s - expected) > 1e-9:
                        return False
    return True
