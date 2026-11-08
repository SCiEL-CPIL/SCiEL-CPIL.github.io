C = A @ B
Cf = np.zeros((n, p))
for ii in range(n):
    for kk in range(p):
        acc = 0.0
        for cc in range(U.shape[1]):
            for jj in range(m):
                acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
        Cf[ii, kk] = acc
