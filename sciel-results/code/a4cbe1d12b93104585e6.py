    A = np.zeros((n * m * p, p * n * r))
    ...
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    col = (k * n + i) * r + c
                    A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                eq_idx += 1
    