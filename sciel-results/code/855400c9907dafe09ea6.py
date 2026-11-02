def _target(n, m, p):
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T[i, j, k] = 1.0
    return T
