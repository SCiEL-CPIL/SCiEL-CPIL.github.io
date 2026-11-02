import time

def construct():
    start_time = time.time()
    # ... construction ...
    if time.time() - start_time > 140:
        # Fallback to schoolbook if something went wrong (should not happen)
        n, m, p = 3, 3, 3
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
