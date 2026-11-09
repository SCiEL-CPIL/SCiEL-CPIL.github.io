import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 6]
_DEADLINE_SECONDS = 110  # safety margin

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def residual_full(x, R):
        """Residual for full 3x3x6 decomposition."""
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:].reshape(p*n, R)
        
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        # T_calc[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        T_target = np.ones((n, m, p))
        
        return (T_calc - T_target).flatten()

    # Try ranks from 46 downward (46 = 2*23 is the trivial bound)
    target_ranks = [46, 45, 44, 43, 42, 41, 40]
    
    best_result = None
    best_cost = np.inf
    best_rank = None
    
    for R in target_ranks:
        if time.time() - t0 > _DEADLINE_SECONDS:
            break
            
        n_vars = 3 * (n*m) * R
        # Use multiple restarts with different strategies
        n_restarts = 5 if R >= 44 else 8
        
        for restart in range(n_restarts):
            if time.time() - t0 > _DEADLINE_SECONDS:
                    break
                
            # Different initialization strategies
            if restart == 0:
                # Random
                x0 = np.random.randn(n_vars)
            elif restart == 1:
                # Sparse random
                x0 = np.random.choice([-1, 0, 1], size=n_vars)
            elif restart == 2:
                # Based on schoolbook structure
                x0 = np.zeros(n_vars)
                U_part = x0[:n*m*R].reshape(n*m, R)
                V_part = x0[n*m*R:2*n*m*R].reshape(m*p, R)
                W_part = x0[2*n*m*R:].reshape(p*n, R)
                # Initialize with a few schoolbook-like terms
                for r in range(min(20, R)):
                    i = (r % n)
                    j = (r // n) % m
                    k1 = (r // (n*m)) % p
                    U_part[i*m+j, r] = 1
                    V_part[j*p+k1, r] = 1
                    W_part[k1*n+i, r] = 1
            else:
                # Perturbed random
                x0 = np.random.randn(n_vars) * 0.5
                
            try:
                res = least_squares(residual_full, x0, args=(R,), method='trf', x_scale='jac', max_nfev=200)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_result = res.x
                    best_rank = R
            except Exception:
                pass

    if best_result is not None and best_cost < 1e-8:
        R = best_rank
        U = best_result[:n*m*R].reshape(n*m, R)
        V = best_result[n*m*R:2*n*m*R].reshape(m*p, R)
        W = best_result[2*n*m*R:].reshape(p*n, R)
        
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        U_t = U_rounded.reshape(n, m, R)
        V_t = V_rounded.reshape(m, p, R)
        W_t = W_rounded.reshape(p, n, R)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        T_target = np.ones((n, m, p))
        
        if np.allclose(T_calc, T_target, atol=1e-9):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
