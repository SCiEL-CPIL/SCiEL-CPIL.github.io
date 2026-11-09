    for i in range(n):
        for j in range(m):
            for k in range(4, p): # k goes 4, 5, 6
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
