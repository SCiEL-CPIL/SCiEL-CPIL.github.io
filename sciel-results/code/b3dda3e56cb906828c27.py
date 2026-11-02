"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build positive and single-mismatch constraints
    constraints = []  # (U_idx, V_idx, W_idx, value)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_idx = i * m + j
                V_idx = j * p + k
                W_idx = k * n + i
                constraints.append((U_idx, V_idx, W_idx, 1.0))
                
                # Single mismatch in j
                for j2 in range(m):
                    if j2 != j:
                        V_idx2 = j2 * p + k
                        constraints.append((U_idx, V_idx2, W_idx, 0.0))
                
                # Single mismatch in i
                for i2 in range(n):
                    if i2 != i:
                        W_idx2 = k * n + i2
                        constraints.append((U_idx, V_idx, W_idx2, 0.0))
                
                # Single mismatch in k
                for k2 in range(p):
                    if k2 != k:
                        W_idx2 = k2 * n + i
                        constraints.append((U_idx, V_idx, W_idx2, 0.0))
    
    n_eq = len(constraints)
    
    def residuals(params, rank):
        U = params[:nm * rank].reshape(nm, rank)
        V = params[nm * rank:(nm + mp) * rank].reshape(mp, rank)
        W = params[(nm + mp) * rank:].reshape(np_, rank)
        
        res = np.zeros(n_eq)
        for idx, (u, v, w, val) in enumerate(constraints):
            res[idx] = np.dot(U[u], V[v] * W[w]) - val
        return res
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    for rank in range(12, 25):
        if time.time() > deadline:
            break
            
        for restart in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + rank * 100 + restart)
            U_init = np.random.randn(nm, rank) * 0.5
            V_init = np.random.randn(mp, rank) * 0.5
            W_init = np.random.randn(np_, rank) * 0.5
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, params_init, args=(rank,), 
                                      method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
            if result.cost < 1e-10:
                U = result.x[:nm * rank].reshape(nm, rank)
                V = result.x[nm * rank:(nm + mp) * rank].reshape(mp, rank)
                W = result.x[(nm + mp) * rank:].reshape(np_, rank)
                
                # Verify full tensor
                full_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for i2 in range(n):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                        actual = np.dot(U[i*m+j], V[j2*p+k2] * W[k2*n+i2])
                                        if abs(actual - expected) > 1e-8:
                                            full_ok = False
                                            break
                                    if not full_ok:
                                        break
                                if not full_ok:
                                    break
                            if not full_ok:
                                break
                        if not full_ok:
                            break
                    if not full_ok:
                        break
                if not full_ok:
                    continue
                
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(np_)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
