"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.flatten()
    
    def make_residual(r):
        def residual(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = x[nm*r+mp*r:].reshape(pn, r)
            R = np.einsum('ad,bd,cd->abc', U, V, W)
            return (R - T).flatten()
        return residual
    
    def schoolbook_init(r):
        U0 = np.zeros((nm, r))
        V0 = np.zeros((mp, r))
        W0 = np.zeros((pn, r))
        for c in range(r):
            i = c // (m*p)
            rem = c % (m*p)
            j = rem // p
            k = rem % p
            U0[i*m+j, c] = 1.0
            V0[j*p+k, c] = 1.0
            W0[k*n+i, c] = 1.0
        return np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    best_rank = n*m*p
    best_params = schoolbook_init(best_rank)
    
    deadline = time.time() + 140
    
    for r in range(best_rank - 1, 0, -1):
        if time.time() > deadline:
            break
        
        residual_fn = make_residual(r)
        x0 = schoolbook_init(r)
        
        # Try schoolbook init with perturbation
        for attempt in range(10):
            if time.time() > deadline:
                break
            if attempt == 0:
                x_try = x0.copy()
            else:
                np.random.seed(attempt * 7919 + r * 31)
                x_try = x0 + np.random.randn(len(x0)) * 0.1
            
            try:
                result = least_squares(residual_fn, x_try, method='lm', max_nfev=10000)
                if result.cost < 1e-12:
                    best_rank = r
                    best_params = result.x
                    break
            except Exception:
                pass
        
        if best_rank == r:
            # Found solution at this rank, try lower
            continue
        else:
            # Try a few random inits for this rank
            for attempt in range(5):
                if time.time() > deadline:
                    break
                np.random.seed(attempt * 104729 + r * 17)
                x_try = np.random.randn(nm*r + mp*r + pn*r) * 0.5
                try:
                    result = least_squares(residual_fn, x_try, method='lm', max_nfev=10000)
                    if result.cost < 1e-12:
                        best_rank = r
                        best_params = result.x
                        break
                except Exception:
                    pass
            if best_rank == r:
                continue
            else:
                # Can't achieve this rank, stop
                break
    
    # Extract U, V, W from best_params
    r = best_rank
    U = best_params[:nm*r].reshape(nm, r)
    V = best_params[nm*r:nm*r+mp*r].reshape(mp, r)
    W = best_params[nm*r+mp*r:].reshape(pn, r)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
