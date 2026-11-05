import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    Y = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    if j == j_prime:
                        Y[i, j, j_prime, k] = 1.0
    Y_flat = Y.flatten()

    def compute_error(params, rank):
        nm = n * m
        mp = m * p
        pn = p * n
        U = params[:nm * rank].reshape(n, m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m, p, rank)
        W = params[nm * rank + mp * rank:].reshape(p, n, rank)
        
        D = np.einsum('ijc,jkc,kic->ijjk', U, V, W)
        err = D.flatten() - Y_flat
        return np.sum(err**2)

    def get_matrices(params, rank):
        nm = n * m
        mp = m * p
        pn = p * n
        U = params[:nm * rank].reshape(n * m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m * p, rank)
        W = params[nm * rank + mp * rank:].reshape(p * n, rank)
        return U, V, W

    def try_rank(rank, max_attempts=10, time_per_attempt=5.0):
        """Try to find a decomposition of given rank using multiple random starts."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B with tighter tolerances
                res = minimize(compute_error, x0, args=(rank,), method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    U_opt, V_opt, W_opt = get_matrices(res.x, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    params_rounded = np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()])
                    err_rounded = compute_error(params_rounded, rank)
                    
                    if err_rounded < 1e-4:
                        return U_rounded, V_rounded, W_rounded
                    elif res.fun < 1e-8:
                        # Accept numerical solution if very accurate
                        return U_opt, V_opt, W_opt
            except Exception:
                continue
        
        return None

    # Start with trivial rank-63 solution
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    # Try to reduce rank
    # Theoretically, rank should be at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But for matrix multiplication, Strassen's algorithm suggests lower bounds.
    # For 3x3 * 3x7, we expect rank around 20-30
    
    for rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_attempts=5, time_per_attempt=8.0)
        
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = rank
            best_U = U_found
            best_V = V_found
            best_W = W_found
            # Continue trying to find even lower rank
            continue
        else:
            # If we can't find a solution for this rank, try next lower rank
            # (it's possible that a lower rank works even if this one doesn't)
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
