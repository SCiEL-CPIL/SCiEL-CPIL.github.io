import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin for safety
    
    # Fallback rank 17 solution
    # (Copied from current program to ensure we always have a valid answer)
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # ... (Insert rank 17 initialization code here) ...
    # I will copy the initialization from the prompt to be safe.
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)
    
    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)
    
    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)
    
    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)
    
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)
    
    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)
    
    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)
    
    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)
    
    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)
    
    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)
    
    best_U, best_V, best_W = U17, V17, W17
    best_rank = 17

    def compute_error_vec(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        error = np.sum((S - 1.0) ** 2)
        return error

    def is_exact(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        S = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.allclose(S, 1.0, atol=1e-9)

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    # Search for rank 16
    target_rank = 16
    found = False
    
    # Try multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        # Initialize
        # Scale initialization to help convergence
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Flatten parameters for optimizer
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*target_rank]
            V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = x[n*m*target_rank + m*p*target_rank :]
            
            U_curr = U_flat.reshape(n, m, target_rank)
            V_curr = V_flat.reshape(m, p, target_rank)
            W_curr = W_flat.reshape(p, n, target_rank)
            
            U_mat = U_curr.reshape(n*m, target_rank)
            V_mat = V_curr.reshape(m*p, target_rank)
            W_mat = W_curr.reshape(p*n, target_rank)
            
            return compute_error_vec(U_mat, V_mat, W_mat, target_rank)
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        except Exception:
            continue
            
        if res.success or res.fun < 1e-6:
            # Extract result
            U_res = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_res = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_res = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            # Snap to half integers
            U_snap, V_snap, W_snap = snap_to_half_integers(U_res, V_res, W_res)
            
            if is_exact(U_snap, V_snap, W_snap, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                found = True
                break
            elif compute_error_vec(U_res, V_res, W_res, target_rank) < 1e-8:
                # If very close but not snapped, keep searching or try local polish
                # For now, if not exact after snap, discard
                pass

    if not found:
        # If rank 16 not found, return rank 17
        pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
