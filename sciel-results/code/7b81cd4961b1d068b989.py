import time
import numpy as np
from scipy.optimize import least_squares
import random

DIMENSIONS = [2, 3, 4]
DEADLINE_MARGIN = 10.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target: S[i, j, k] = 1
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # S[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def is_valid(U, V, W, tol=1e-9):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        err = np.max(np.abs(S - 1.0))
        return err < tol, err

    def snap_half(x):
        return np.round(x * 2.0) / 2.0

    # Schoolbook decomposition (Rank 24)
    rank_school = n * m * p
    U0 = np.zeros((n*m, rank_school))
    V0 = np.zeros((m*p, rank_school))
    W0 = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_school

    # Search for lower ranks
    # Known minimal rank for 2x3x4 is 17.
    for r in range(17, rank_school):
        if time.time() - start_time > 120 - DEADLINE_MARGIN:
            break
            
        found = False
        # Try a few random initializations
        # Use more restarts for lower ranks
        num_restarts = 20 if r < 20 else 10
        
        for attempt in range(num_restarts):
            if time.time() - start_time > 120 - DEADLINE_MARGIN:
                break
                
            # Random initialization
            # Scale to roughly 1/sqrt(r) to avoid huge values initially
            U = np.random.randn(n*m, r) * (1.0 / np.sqrt(r))
            V = np.random.randn(m*p, r) * (1.0 / np.sqrt(r))
            W = np.random.randn(p*n, r) * (1.0 / np.sqrt(r))
            
            # Flatten for optimization
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
            def unflatten(x):
                u_size = n*m*r
                v_size = m*p*r
                w_size = p*n*r
                U_f = x[:u_size].reshape(n*m, r)
                V_f = x[u_size:u_size+v_size].reshape(m*p, r)
                W_f = x[u_size+v_size:].reshape(p*n, r)
                return U_f, V_f, W_f

            def residuals(x):
                U_f, V_f, W_f = unflatten(x)
                U_r = U_f.reshape(n, m, r)
                V_r = V_f.reshape(m, p, r)
                W_r = W_f.reshape(p, n, r)
                S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                return (S - 1.0).flatten()

            x0 = flatten(U, V, W)
            
            try:
                # Use least_squares for robustness
                result = least_squares(residuals, x0, method='lm', max_nfev=500)
                
                if result.success or result.cost < 1e-10:
                    U_opt, V_opt, W_opt = unflatten(result.x)
                    valid, err = is_valid(U_opt, V_opt, W_opt, tol=1e-6)
                    
                    if valid:
                        # Snap to half-integers
                        U_s = snap_half(U_opt)
                        V_s = snap_half(V_opt)
                        W_s = snap_half(W_opt)
                        
                        valid_s, err_s = is_valid(U_s, V_s, W_s)
                        if valid_s:
                            best_U, best_V, best_W = U_s, V_s, W_s
                            best_rank = r
                            found = True
                            break
            except Exception:
                pass

        if found:
            break

    # Return best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
