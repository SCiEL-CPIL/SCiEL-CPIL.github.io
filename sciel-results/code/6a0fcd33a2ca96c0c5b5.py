for j in range(m):   # j in [4], this is the contraction index k in my notation
    for k in range(p):  # k in [8], this is the column index j in my notation
        U[j][c] = 1.0
        V[j * p + k][c] = 1.0
        W[k * n][c] = 1.0
