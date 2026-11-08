import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    T_flat = T.flatten()
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = nm * mp * pn
    
    # Try ranks from 47 down to 40
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Fewer restarts for higher dimensions
        max_restarts = 10 if rank > 45 else 5
        
        for _ in range(max_restarts):
            if time.time() > deadline - 20:
                break
                
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            def residuals(params):
                U_r = params[:nm*rank].reshape(nm, rank)
                V_r = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_r = params[nm*rank+mp*rank:].reshape(pn, rank)
                T_approx = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                return T_approx.flatten() - T_flat
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=500)
                if res.cost < 1e-10:
                    U_opt = res.x[:nm*rank].reshape(nm, rank)
                    V_opt = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_opt = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    T_check = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
                    if np.allclose(T_check, T, atol=1e-5):
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        best_rank = rank
                        found = True
                        break
            except Exception:
                pass
                
        if found:
            continue
        else:
            # If not found, continue to next lower rank?
            # Or stop?
            # Let's continue.
            pass

    if best_U is None:
        rank = nm * mp * pn
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
