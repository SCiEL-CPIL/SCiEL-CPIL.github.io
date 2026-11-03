def _construct_np(n, m, p):
    rank = n * p
    # ...
    for i in range(n):
        for k in range(p):
            c = k * n + i
            for j in range(m):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
            W[k * n + i, c] = 1
