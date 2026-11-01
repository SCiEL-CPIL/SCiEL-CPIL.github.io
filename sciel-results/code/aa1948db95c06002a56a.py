"""Rewritten program to find rank-minimal tensor decomposition for 2x4 x 4x6 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]
DEADLINE = 135.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def residual(params, r):
        """Compute residual for given parameters and rank."""
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:].reshape(pn, r)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res.append(s - 1.0)
        return np.array(res)
    
    # Try to find lower rank decompositions
    for target_rank in range(rank_school - 1, max(rank_school - 8, 35), -1):
        if time.time() - start_time > DEADLINE:
            break
        
        r = target_rank
        # Initialize with schoolbook pattern for first r terms
        init = np.zeros(nm*r + mp*r + pn*r)
        for c in range(r):
            # Map c to (i,j,k)
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            init[i*m+j] = 1
            init[nm*r + j*p+k] = 1
            init[nm*r + mp*r + k*n+i] = 1
        
        try:
            result = least_squares(residual, init, args=(r,), method='lm', max_nfev=1000)
            if time.time() - start_time > DEADLINE:
                break
            if result.success or np.max(np.abs(result.fun)) < 1e-9:
                U = result.x[:nm*r].reshape(nm, r)
                V = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W = result.x[nm*r+mp*r:].reshape(pn, r)
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(s - 1.0) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    break
        except Exception:
            continue
    
    if time.time() - start_time > DEADLINE:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
