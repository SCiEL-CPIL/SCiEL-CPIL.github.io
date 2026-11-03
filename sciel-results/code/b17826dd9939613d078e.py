for i in range(n):
    for j in range(m):
        for k in range(p):
            if j in (0, 1) and k in (0, 1, 2, 3):
                continue
            U[i*m+j][c] = 1
            V[j*p+k][c] = 1
            W[k*n+i][c] = 1
            c += 1
