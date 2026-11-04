import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS

    T = np.zeros((n, m, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            T[i, j, j, :] = 1.0

    def max_residual(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for l in range(p):
                U_i = U[i*m:(i+1)*m, :]
                V_l = V[l::p, :]
                W_il = W[l*n+i, :]
                M = (U_i * W_il) @ V_l.T
                T_il = T[i, :, :, l]
                err = np.max(np.abs(M - T_il))
                if err > max_err:
                    max_err = err
        return max_err

    def alt_minimize(U, V, W, r, max_iter=400, tol=1e-13):
        for it in range(max_iter):
            if time.time() > deadline - 15:
                return False
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((m * p, r))
                    for k in range(m):
                        for l in range(p):
                            M[k * p + l, :] = V[k * p + l, :] * W[l * n + i, :]
                    T_vec = T[i, j, :, :].reshape(m * p)
                    try:
                        U[i * m + j, :] = np.linalg.solve(M.T @ M + 1e-14 * np.eye(r), M.T @ T_vec)
                    except:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, T_vec, rcond=None)[0]
            
            for k in range(m):
                for l in range(p):
                    M = np.zeros((n * m, r))
                    for i in range(n):
                        for j in range(m):
                            M[i * m + j, :] = U[i * m + j, :] * W[l * n + i, :]
                    T_vec = T[:, :, k, l].reshape(n * m)
                    try:
                        V[k * p + l, :] = np.linalg.solve(M.T @ M + 1e-14 * np.eye(r), M.T @ T_vec)
                    except:
                        V[k * p + l, :] = np.linalg.lstsq(M.T, T_vec, rcond=None)[0]
            
            for l in range(p):
                for i in range(n):
                    M = np.zeros((m * m, r))
                    for j in range(m):
                        for k in range(m):
                            M[j * m + k, :] = U[i * m + j, :] * V[k * p + l, :]
                    T_vec = T[i, :, :, l].reshape(m * m)
                    try:
                        W[l * n + i, :] = np.linalg.solve(M.T @ M + 1e-14 * np.eye(r), M.T @ T_vec)
                    except:
                        W[l * n + i, :] = np.linalg.lstsq(M.T, T_vec, rcond=None)[0]
            
            if max_residual(U, V, W, r) < tol:
                return True
        return False

    best_result = None
    best_rank = None

    for target_rank in range(50, 121, 5):
        if time.time() > deadline - 20:
            break
        for trial in range(4):
            if time.time() > deadline - 20:
                break
            np.random.seed(123 + target_rank * 13 + trial * 101)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            if alt_minimize(U, V, W, target_rank):
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                if max_residual(Ur, Vr, Wr, target_rank) < 1e-9:
                    best_rank = target_rank
                    best_result = (Ur, Vr, Wr)
                    break
        if best_result is not None:
            break

    if best_result is not None:
        Ur, Vr, Wr = best_result
        return {"U": [[float(x) for x in row] for row in Ur],
                "V": [[float(x) for x in row] for row in Vr],
                "W": [[float(x) for x in row] for row in Wr]}

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
