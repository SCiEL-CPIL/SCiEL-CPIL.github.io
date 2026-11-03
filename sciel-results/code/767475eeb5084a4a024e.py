for i in range(n):      # i: 0..2 (row of A, row of C)
    for j in range(m):  # j: 0..2 (col of A, row of B)
        for k in range(p): # k: 0..2 (col of B, col of C)
            U[i * m + j][column] = 1   # A[i,j]
            V[j * p + k][column] = 1   # B[j,k]
            W[k * n + i][column] = 1   # C[i,k] but indexed as k*n+i
            column += 1
