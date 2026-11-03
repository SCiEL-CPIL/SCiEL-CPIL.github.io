    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    total = 0.0
                    for r in range(R):
                        total += W[k * n + i][r] * U[i * m + j][r] * V[jp * p + k][r]
                    expected = 1.0 if j == jp else 0.0
                    if abs(total - expected) > 1e-9:
                        return False
    