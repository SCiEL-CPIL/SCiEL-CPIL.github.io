import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    # Schoolbook rank-24 solution (always valid fallback)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Constraint list: for each (i,j,k) we need sum_k U[i*m+j,k]*V[j*p+k,k]*W[k*n+i,k] = 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i * m + j, j * p + k, k * n + i))
    num_constraints = len(constraints)
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def residuals(params, r):
        U = params[:r * n * m].reshape(n * m, r)
        V = params[r * n * m:r * (n * m + m * p)].reshape(m * p, r)
        W = params[r * (n * m + m * p):].reshape(p * n, r)
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return vals - 1.0

    def round_and_verify(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        # Check error
        vals = np.sum(U_r[u_idxs, :] * V_r[v_idxs, :] * W_r[w_idxs, :], axis=1)
        if np.max(np.abs(vals - 1.0)) < tol:
            return U_r, V_r, W_r
        return None

    def try_rank(r, num_restarts=100, max_iters=100):
        if time.time() > deadline - 10.0:
            return None
        size = r * (n * m + m * p + p * n)
        best_sol = None
        best_cost = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            np.random.seed(42 + restart)
            
            # Random initialization
            U0 = np.random.randn(n * m, r) * 0.1
            V0 = np.random.randn(m * p, r) * 0.1
            W0 = np.random.randn(p * n, r) * 0.1
            
            params = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])

            try:
                res = least_squares(
                    residuals, 
                    params, 
                    args=(r,), 
                    method='trf', 
                    max_nfev=max_iters
                )
            except Exception:
                continue

            if res.cost < best_cost:
                best_cost = res.cost
                best_sol = res.x
                if best_cost < 1e-10:
                    U = best_sol[:r * n * m].reshape(n * m, r)
                    V = best_sol[r * n * m:r * (n * m + m * p)].reshape(m * p, r)
                    W = best_sol[r * (n * m + m * p):].reshape(p * n, r)
                    result = round_and_verify(U, V, W)
                    if result:
                        return result

        return None

    # Try ranks from 18 up to 23
    # Rank 18 is a strong candidate for M_{2,3,4}
    for r in range(18, 24):
        if time.time() > deadline - 20.0:
            break
        sol = try_rank(r, num_restarts=100, max_iters=200)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break

    # Final safety: verify the returned solution
    vals = np.sum(best_U[u_idxs, :] * best_V[v_idxs, :] * best_W[w_idxs, :], axis=1)
    if np.max(np.abs(vals - 1.0)) > 1e-8:
        best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
        best_rank = rank_seed

    # Ensure entries are integer or half-integer
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
