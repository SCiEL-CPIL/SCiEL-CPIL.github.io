import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    rank_schoolbook = n * m * p

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Try to find a lower-rank decomposition
    for R in [35, 34, 33, 32]:
        if time.time() > deadline - 20:
            break

        # Initialize with schoolbook truncated to R columns
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= R:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

        def flatten(x, y, z):
            return np.concatenate([x.flatten(), y.flatten(), z.flatten()])

        def unflatten(v):
            U = v[:n * m * R].reshape(n * m, R)
            V = v[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
            W = v[n * m * R + m * p * R:].reshape(p * n, R)
            return U, V, W

        def residual(v):
            if time.time() > deadline - 10:
                return np.ones(n * m * m * p * p * n)
            U, V, W = unflatten(v)
            T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            return (T_comp - T).flatten()

        x0 = flatten(U, V, W)

        try:
            result = least_squares(residual, x0, method='lm', max_nfev=500)
            if result.cost < 1e-8:
                U, V, W = unflatten(result.x)
                T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                if np.max(np.abs(T_comp - T)) < 1e-7:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass

    # Fall back to schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
