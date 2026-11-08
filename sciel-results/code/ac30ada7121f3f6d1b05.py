u_ar0 = 0 * m + 2  # = 2
u_ar1 = 1 * m + 2  # = 5
for k in range(p):
    v_bk = 2 * p + k  # = 18+k
    w_0k = k * n + 0  # = 2k
    w_1k = k * n + 1  # = 2k+1
    U[u_ar0][col] += 1
    V[v_bk][col] += 1
    W[w_0k][col] += 1
    U[u_ar1][col+1] += 1
    V[v_bk][col+1] += 1
    W[w_1k][col+1] += 1
