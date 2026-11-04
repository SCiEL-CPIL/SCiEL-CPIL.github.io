import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    start_time = time.time()
    deadline = start_time + 100

    # Try to find a lower rank decomposition with a tight time limit
    best_result = None
    search_deadline = start_time + 5

    for R in range(25, min(rank_school, 40)):
        if time.time() > search_deadline:
            break

        for trial in range(3):
            if time.time() > search_deadline:
                break

            np.random.seed(trial * 1000 + R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)

            U = np.zeros((n * m, R))
            residual = 0.0

            for a in range(n * m):
                M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, R)
                T_a = np.zeros(m * p * p * n)

                i = a // m
                j = a % m
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    T_a[b * p * n + c] = 1.0

                try:
                    U[a, :] = np.linalg.lstsq(M_a, T_a, rcond=None)[0]
                except:
                    continue

                pred = M_a @ U[a, :]
                residual += np.sum((pred - T_a) ** 2)

            if residual < 1e-8:
                all_ok = True
                for mat in [U, V, W]:
                    for row in mat:
                        for val in row:
                            nearest = round(val * 2) / 2
                            if abs(val - nearest) > 1e-4:
                                all_ok = False
                                break
                        if not all_ok:
                            break
                    if not all_ok:
                        break

                if all_ok:
                    best_result = {
                        "U": [[round(val * 2) / 2 for val in row] for row in U],
                        "V": [[round(val * 2) / 2 for val in row] for row in V],
                        "W": [[round(val * 2) / 2 for val in row] for row in W]
                    }
                    break

        if best_result is not None:
            break

    if best_result is not None:
        return best_result

    # Return schoolbook
    rank = rank_school
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
