import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    n, m, p = 2, 3, 8

    def schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def tensor_residual(U, V, W, n, m, p):
        R = U.shape[1]
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    residual[i, j, k] = s - 1.0
        return residual

    def verify(U, V, W, n, m, p, tol=1e-6):
        res = tensor_residual(U, V, W, n, m, p)
        return np.max(np.abs(res)) < tol

    U0, V0, W0 = schoolbook(n, m, p)
    assert verify(U0, V0, W0, n, m, p), "Schoolbook failed"

    best_rank = n * m * p
    best_decomp = (U0.copy(), V0.copy(), W0.copy())

    # Try to reduce rank using nonlinear optimization
    for target_rank in range(best_rank - 1, 35, -1):
        if time.time() > deadline - 15:
            break

        found = False
        for trial in range(30):
            if time.time() > deadline - 15:
                break

            # Initialize with random subset of schoolbook terms
            idx = np.random.choice(best_rank, target_rank, replace=False)
            U_cur = U0[:, idx].copy()
            V_cur = V0[:, idx].copy()
            W_cur = W0[:, idx].copy()

            # Perturb to escape local minima
            U_cur += np.random.randn(*U_cur.shape) * 0.5
            V_cur += np.random.randn(*V_cur.shape) * 0.5
            W_cur += np.random.randn(*W_cur.shape) * 0.5

            def residuals(params):
                R = target_rank
                U_flat = params[:n * m * R]
                V_flat = params[n * m * R:n * m * R + m * p * R]
                W_flat = params[n * m * R + m * p * R:]

                U_mat = U_flat.reshape(n * m, R)
                V_mat = V_flat.reshape(m * p, R)
                W_mat = W_flat.reshape(p * n, R)

                res = tensor_residual(U_mat, V_mat, W_mat, n, m, p)
                return res.flatten()

            params0 = np.concatenate([
                U_cur.flatten(),
                V_cur.flatten(),
                W_cur.flatten()
            ])

            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=200)

                if result.cost < 1e-8:
                    R = target_rank
                    U_opt = result.x[:n * m * R].reshape(n * m, R)
                    V_opt = result.x[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
                    W_opt = result.x[n * m * R + m * p * R:].reshape(p * n, R)

                    if verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        best_rank = target_rank
                        best_decomp = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                        found = True
                        break
            except Exception:
                continue

        if not found:
            continue

    U_best, V_best, W_best = best_decomp
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
