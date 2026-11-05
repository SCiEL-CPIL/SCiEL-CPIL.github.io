import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Precompute the target tensor
    Y = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Y[i, j, j, k] = 1.0
    Y_flat = Y.flatten()

    def compute_error_fast(params, rank, nm, mp, pn):
        U = params[:nm * rank].reshape(n, m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m, p, rank)
        W = params[nm * rank + mp * rank:].reshape(p, n, rank)
        
        # D[i,j,j',k] = sum_c U[i,j,c] * V[j',k,c] * W[k,i,c]
        D = np.einsum('ijc,jkc,kic->ijjk', U, V, W, optimize=True)
        err = D.flatten() - Y_flat
        return np.sum(err**2)

    def get_matrices(params, rank, nm, mp, pn):
        U = params[:nm * rank].reshape(n * m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m * p, rank)
        W = params[nm * rank + mp * rank:].reshape(p * n, rank)
        return U, V, W

    def try_rank(rank, max_attempts=8, time_per_attempt=3.0):
        """Try to find a decomposition of given rank using multiple random starts."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 8:
                break
            
            # Random initialization with some structure
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(compute_error_fast, x0, args=(rank, nm, mp, pn), method='L-BFGS-B', 
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
                
                if res.fun < 1e-4:
                    U_opt, V_opt, W_opt = get_matrices(res.x, rank, nm, mp, pn)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    params_rounded = np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()])
                    err_rounded = compute_error_fast(params_rounded, rank, nm, mp, pn)
                    
                    if err_rounded < 1e-3:
                        return U_rounded, V_rounded, W_rounded
                    elif res.fun < 1e-6:
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

    nm = n * m
    mp = m * p
    pn = p * n

    # Try to reduce rank - start from reasonable estimates
    # For 3x3 * 3x7, Strassen-like bounds suggest rank ~ 20-30
    # Try ranks from 30 down to 21
    for rank in range(30, 20, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_attempts=6, time_per_attempt=3.0)
        
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = rank
            best_U = U_found
            best_V = V_found
            best_W = W_found
            break  # Found a solution, stop here

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
