import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135

    def check_solution(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    min_rank = max(n*m, m*p, p*n)
    max_rank = n * m * p

    found = False
    best_U = best_V = best_W = None
    best_r = max_rank

    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break

        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            np.random.seed(777 + attempt * 13)
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3

            for iteration in range(3000):
                if time.time() > deadline:
                    break

                for i in range(n):
                    for j in range(m):
                        C = np.array([np.sum(V[j*p + k, r2] * W[k*n + i, r2] for k in range(p)) for r2 in range(r)])
                        if np.linalg.norm(C) > 1e-14:
                            U[i*m+j, :] = C / np.dot(C, C)
                        U[i*m+j, :] = np.round(2 * U[i*m+j, :]) / 2.0

                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, r2] * W[k*n+i, r2] for r2 in range(r)] for i in range(n)])
                        b = np.ones(n)
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except:
                            V[j*p+k, :] = np.zeros(r)
                        V[j*p+k, :] = np.round(2 * V[j*p+k, :]) / 2.0

                for k in range(p):
                    for i in range(n):
                        A = np.array([[U[i*m+j, r2] * V[j*p+k, r2] for r2 in range(r)] for j in range(m)])
                        b = np.ones(m)
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except:
                            W[k*n+i, :] = np.zeros(r)
                        W[k*n+i, :] = np.round(2 * W[k*n+i, :]) / 2.0

                if check_solution(U, V, W, r):
                    found = True
                    best_U, best_V, best_W = U, V, W
                    best_r = r
                    break

        if found:
            break

    if not found:
        best_r = n * m * p
        best_U = np.zeros((n*m, best_r))
        best_V = np.zeros((m*p, best_r))
        best_W = np.zeros((p*n, best_r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1

    U_list = [[float(best_U[i, j]) for j in range(best_r)] for i in range(n*m)]
    V_list = [[float(best_V[i, j]) for j in range(best_r)] for i in range(m*p)]
    W_list = [[float(best_W[i, j]) for j in range(best_r)] for i in range(p*n)]

    return {"U": U_list, "V": V_list, "W": W_list}
