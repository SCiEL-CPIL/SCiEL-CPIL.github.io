C_computed = np.zeros((n, p))
for r in range(rank):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C_computed[i, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
