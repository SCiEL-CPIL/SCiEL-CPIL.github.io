import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 6

    def try_rank(rank, seed=0, max_iter=30000):
        np.random.seed(seed)

        def objective(params):
            U = params[:n*m*rank].reshape(n*m, rank)
            V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            return err

        # Initialize: schoolbook for first min(rank,54) cols, random for rest
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c in range(min(rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        for c in range(n*m*p, rank):
            U[:, c] = np.random.randn(n*m) * 0.2
            V[:, c] = np.random.randn(m*p) * 0.2
            W[:, c] = np.random.randn(p*n) * 0.2

        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        result = minimize(objective, params, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12})
        if result.fun < 1e-8:
            U_f = result.x[:n*m*rank].reshape(n*m, rank)
            V_f = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_f = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            U_r = np.round(U_f * 2) / 2
            V_r = np.round(V_f * 2) / 2
            W_r = np.round(W_f * 2) / 2
            max_err = max(abs(np.dot(U_r[i*m+j,:], V_r[j*p+k,:] * W_r[k*n+i,:]) - 1.0)
                          for i in range(n) for j in range(m) for k in range(p))
            if max_err < 1e-6:
                return U_r, V_r, W_r
        return None

    for rank in [46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23]:
        for seed in [0, 1, 2]:
            res = try_rank(rank, seed=seed, max_iter=20000)
            if res is not None:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
