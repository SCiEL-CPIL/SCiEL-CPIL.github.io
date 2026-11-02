import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Try ranks from 9 to 17
    for R in range(9, 18):
        if time.time() > deadline - 10:
            break

        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None

        # Try multiple random restarts with different seeds
        for restart in range(100):
            if time.time() > deadline - 5:
                break

            np.random.seed(12345 + restart)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5

            # ALS iterations
            for it in range(150):
                if time.time() > deadline - 2:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = W[k * n + i, :] * V[j * p + k, :]
                        try:
                            U_vec, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[i * m + j, :] = U_vec
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = W[k * n + i, :] * U[i * m + j, :]
                        try:
                            V_vec, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[j * p + k, :] = V_vec
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W_vec, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[k * n + i, :] = W_vec
                        except:
                            pass

                # Check error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(W[k * n + i, :], U[i * m + j, :] * V[j * p + k, :])
                            error += (val - 1) ** 2

                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()

                if best_error < 1e-8:
                    break

            if best_error < 1e-8:
                # Found a solution. Try to round to integers/half-integers
                # Scale and round
                for scale in [1, 2, 4, 0.5, 0.25, 8, 0.125]:
                    U_r = np.round(best_U * scale) / scale
                    V_r = np.round(best_V * scale) / scale
                    W_r = np.round(best_W * scale) / scale

                    # Verify
                    error_r = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(W_r[k * n + i, :], U_r[i * m + j, :] * V_r[j * p + k, :])
                                error_r += (val - 1) ** 2

                    if error_r < 1e-8:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }

                # If rounding failed, try to find a rational approximation
                # This is more complex, so we'll fall back to the schoolbook
                break

    # Fallback to schoolbook
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
