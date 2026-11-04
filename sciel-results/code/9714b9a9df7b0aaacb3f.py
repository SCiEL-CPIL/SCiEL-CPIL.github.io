def verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    # compute T(e_{i,j}, e_{jp,k})
                    C = [0]*(p*n)
                    for r in range(rank):
                        uv = U[i*m+j][r] * V[jp*p+k][r]
                        if uv:
                            for pos in range(p*n):
                                C[pos] += uv * W[pos][r]
                    # expected
                    exp = [0]*(p*n)
                    if j == jp:
                        exp[k*n+i] = 1
                    if C != exp:
                        return False
    return True
