import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    T_flat = T.flatten()
    
    def compute_error(params, rank):
        # params is a flat vector of size (n*m + m*p + p*n) * rank
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        
        U = params[:U_size].reshape((n*m, rank))
        V = params[U_size:U_size+V_size].reshape((m*p, rank))
        W = params[U_size+V_size:].reshape((p*n, rank))
        
        # Compute T_rec
        # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        T_rec_flat = T_rec.flatten()
        
        err = T_rec_flat - T_flat
        return err

    def objective(params, rank):
        err = compute_error(params, rank)
        return np.sum(err**2)

    def check_decomposition(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if entries are close to 0 or 1
        # And if U, V, W are close to half-integers
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        # Check half-integer property
        U_check = 2 * U
        V_check = 2 * V
        W_check = 2 * W
        if not np.allclose(U_check, np.round(U_check), atol=1e-6):
            return False
        if not np.allclose(V_check, np.round(V_check), atol=1e-6):
            return False
        if not np.allclose(W_check, np.round(W_check), atol=1e-6):
            return False
        return True

    def snap_to_half_integer(X):
        return 0.5 * np.round(2 * X)

    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Try ranks from 18 up to 23
    # We know schoolbook is 24.
    # If 18 is the true rank, we should find it.
    for rank in range(18, 24):
        if time.time() - start_time > time_limit:
            break
            
        success = False
        # Multiple restarts
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with small random values
            # Or initialize with schoolbook-like structure?
            # Random is safer for general search
            np.random.seed(42 + restart)
            
            U0 = np.random.randn(n*m, rank) * 0.1
            V0 = np.random.randn(m*p, rank) * 0.1
            W0 = np.random.randn(p*n, rank) * 0.1
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use L-BFGS-B or BFGS
            # BFGS is good for medium sized problems
            try:
                res = minimize(objective, params0, args=(rank,), method='BFGS', 
                               options={'maxiter': 500, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                # Extract U, V, W
                U_size = n * m * rank
                V_size = m * p * rank
                U = res.x[:U_size].reshape((n*m, rank))
                V = res.x[U_size:U_size+V_size].reshape((m*p, rank))
                W = res.x[U_size+V_size:].reshape((p*n, rank))
                
                # Snap to half-integers
                U_snap = snap_to_half_integer(U)
                V_snap = snap_to_half_integer(V)
                W_snap = snap_to_half_integer(W)
                
                if check_decomposition(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                else:
                    # Maybe the real solution is exact but snapping failed?
                    # Or maybe the solution found is not the integer one.
                    # Try snapping with slightly different tolerance or check if original was close
                    if check_decomposition(U, V, W):
                         # This shouldn't happen if check_decomposition requires half-integers
                         # But if the real solution IS half-integer, snapping should work.
                         # If snapping fails, maybe the solution is not half-integer?
                         # But the problem REQUIRES half-integer entries.
                         # So we must find a half-integer solution.
                         pass

        # If rank 'rank' not found, try next rank
    
    # Fallback to schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
