        for i in range(n):
            for k in range(p):
                for j in range(m):
                    r = i * p * m + k * m + j
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
