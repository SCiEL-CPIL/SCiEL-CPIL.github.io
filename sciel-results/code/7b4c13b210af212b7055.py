for i in range(n):
    for k in range(p):
        for j in range(m):
            for l in range(m):
                total = Fraction(0)
                for r in range(rank):
                    total += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                expected = Fraction(1) if j == l else Fraction(0)
                if total != expected:
                    return False
