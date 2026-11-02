import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 3, 3, 5

    for rank in [25]:
        for trial in range(300):
            if time.time() > deadline:
                break

            np.random.seed(trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)

            for it in range(2000):
                if time.time() > deadline:
                    break

                # Solve for U (fix V, W)
                for i in range(n):
                    for j in range(m):
                        V_sub = V[j * p:(j + 1) * p, :]
                        W_sub = W[i::n, :p]
                        A = V_sub * W_sub
                        b = np.ones(p)
                        U[i * m + j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                # Solve for V (fix U, W)
                for j in range(m):
                    for k in range(p):
                        U_sub = U[:, j]
                        W_sub = W[k * n:(k + 1) * n, :]
                        A = U_sub * W_sub
                        b = np.ones(n)
                        V[j * p + k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                # Solve for W (fix U, V)
                for i in range(n):
                    for k in range(p):
                        U_sub = U[i * m:(i + 1) * m, :]
                        V_sub = V[k * p:(k + 1) * p, :]
                        A = U_sub * V_sub
                        b = np.ones(m)
                        W[k * n + i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                # Check error every 10 iterations
                if it % 10 == 0:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                err += (val - 1.0) ** 2
                    if err < 1e-8:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook decomposition
    rank = n * m * p
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
