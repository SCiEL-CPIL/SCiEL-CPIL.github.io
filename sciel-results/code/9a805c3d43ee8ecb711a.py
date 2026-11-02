import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    for rank in range(120, 89, -1):
        if time.time() > deadline - 10:
            break
        for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
            if time.time() > deadline - 10:
                break
            result = try_rank_als(n, m, p, rank, seed)
            if result is not None:
                return result
    
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
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

def try_rank_als(n, m, p, rank, seed):
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(100):
        W_new = np.zeros_like(W)
        for i in range(n):
            for k in range(p):
                A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                b = np.ones(m)
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                W_new[k * n + i, :] = sol
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                b = np.ones(n)
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                V_new[j * p + k, :] = sol
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                b = np.ones(p)
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                U_new[i * m + j, :] = sol
        
        U, V, W = U_new, V_new, W_new
        
        err = compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-8:
        # Round to nearest 0.5
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        
        err_r = compute_error(U_r, V_r, W_r, n, m, p)
        if err_r < 1e-10:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            # Return unrounded if rounding fails
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def compute_error(U, V, W, n, m, p):
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err
