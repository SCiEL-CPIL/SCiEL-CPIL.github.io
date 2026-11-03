        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > 1e-9:
                        return _schoolbook(n, m, p)
