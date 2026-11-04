import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = nm * p # 45
    deadline = time.time() + 140  # Safety margin before 150s limit

    def tensor_residuals(U, V, W):
        # Compute the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # and return the residual T - 1
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i*m+j]
                    v = V[j*p+k]
                    w = W[k*n+i]
                    res[i,j,k] = np.dot(u, v * w) - 1.0
        return res.flatten()

    def verify_exact(U, V, W):
        # Check if the decomposition is exact with integer/half-integer entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i*m+j]
                    v = V[j*p+k]
                    w = W[k*n+i]
                    val = np.dot(u, v * w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def round_to_half(x):
        # Round to nearest half-integer
        return np.round(x * 2) / 2

    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank

    # Try to find lower ranks
    for target_rank in range(schoolbook_rank - 1, 30, -1):
        if time.time() > deadline:
            break

        # Multiple restarts
        for restart in range(10):
            if time.time() > deadline:
                break

            # Initialize with small random values
            rng = np.random.default_rng(restart)
            U_init = rng.normal(size=(nm, target_rank)) * 0.1
            V_init = rng.normal(size=(mp, target_rank)) * 0.1
            W_init = rng.normal(size=(pn, target_rank)) * 0.1

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def objective(x):
                U = x[:nm*target_rank].reshape(nm, target_rank)
                V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                res = tensor_residuals(U, V, W)
                return np.sum(res**2)

            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-12})

                if res.fun < 1e-10:
                    U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                    V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)

                    # Round to half-integers
                    U_rounded = round_to_half(U_sol)
                    V_rounded = round_to_half(V_sol)
                    W_rounded = round_to_half(W_sol)

                    # Verify the rounded solution
                    if verify_exact(U_rounded, V_rounded, W_rounded):
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        best_rank = target_rank
                        break  # Found a valid solution at this rank, try lower

            except Exception:
                pass

    if best_U is None:
        # Return schoolbook
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
        best_U, best_V, best_W = U, V, W

    return {"U": best_U, "V": best_V, "W": best_W}
