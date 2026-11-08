for i in range(n):
    for j in range(m):
        for k in range(p):
            val = 0.0
            ai = i * m + j
            bi = j * p + k
            wi = k * n + i
            for l in range(r):
                val += U[ai][l] * V[bi][l] * W[wi][l]
            if abs(val - 1.0) > 1e-9:
                return False
