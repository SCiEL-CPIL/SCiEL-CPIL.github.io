# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin

    # Schoolbook decomposition of rank n*m*p
    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    # Precompute target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def evaluate(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        return np.einsum('ar,br,cr->abc', U, V, W)

    def residual(params, R):
        size_U = n * m * R
        size_V = m * p * R
        U = params[:size_U].reshape((n * m, R))
        V = params[size_U:size_U + size_V].reshape((m * p, R))
        W = params[size_U + size_V:].reshape((p * n, R))
        T_calc = evaluate(U, V, W)
        return (T_calc - T_target).flatten()

    def try_rank(R, num_restarts=5):
        if time.time() > deadline - 10:
            return None
            
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        best_loss = float('inf')
        best_params = None
        
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                break
            # Random initialization
            params = np.random.randint(-2, 3, size=total_size).astype(float)
            
            try:
                sol = least_squares(
                    lambda x: residual(x, R),
                    params,
                    method='lm',
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    max_nfev=2000
                )
                if sol.fun < best_loss:
                    best_loss = sol.fun
                    best_params = sol.x
            except Exception:
                pass
                
        if best_params is None:
            return None
            
        U = best_params[:size_U].reshape((n * m, R))
        V = best_params[size_U:size_U + size_V].reshape((m * p, R))
        W = best_params[size_U + size_V:].reshape((p * n, R))
        
        # Check if close to integers/half-integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        U_round = np.round(U2)
        V_round = np.round(V2)
        W_round = np.round(W2)
        
        U_check = U_round / 2.0
        V_check = V_round / 2.0
        W_check = W_round / 2.0
        
        err = np.sum((evaluate(U_check, V_check, W_check) - T_target)**2)
        
        if err < 1e-6:
            return {
                "U": [[float(x) for x in row] for row in U_check],
                "V": [[float(x) for x in row] for row in V_check],
                "W": [[float(x) for x in row] for row in W_check]
            }
            
        return None

    # Try ranks from best_rank - 1 down to best_rank - 15
    for R in range(best_rank - 1, best_rank - 16, -1):
        if time.time() > deadline - 5:
            break
        result = try_rank(R, num_restarts=3)
        if result is not None:
            return result

    return best
# EVOLVE-BLOCK-END
