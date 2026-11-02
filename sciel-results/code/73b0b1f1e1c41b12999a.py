"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_set = set(target_indices)
    T_target = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Schoolbook Rank 50 Solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Helper to check and snap
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        # Reconstruct
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
            
        return {
            "U": U_snap.tolist(),
            "V": V_snap.tolist(),
            "W": W_snap.tolist()
        }

    # Optimization function for a given rank r and initial guess
    def try_optimize(r, U_init, V_init, W_init, max_iter=50):
        if time.time() > deadline:
            return None
            
        size_U = (n * m) * r
        size_V = (m * p) * r
        size_W = (p * n) * r
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def loss(params):
            U_p = params[:size_U].reshape((n * m), r)
            V_p = params[size_U:size_U + size_V].reshape((m * p), r)
            W_p = params[size_U + size_V:size_U + size_V + size_W].reshape((p * n), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err
        
        from scipy.optimize import minimize
        try:
            # Use L-BFGS-B, it's fast for this size
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'maxfun': 100})
        except Exception:
            return None
            
        if res.fun < 1e-6:
            U_sol = res.x[:size_U].reshape((n * m), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
            W_sol = res.x[size_U + size_V:].reshape((p * n), r)
            
            result = check_and_snap(U_sol, V_sol, W_sol, r)
            if result:
                return result
        return None

    # Strategy: Try to find Rank 49 by perturbing the Rank 50 solution.
    # We remove one term from the schoolbook solution and try to optimize the rest.
    # This is a "hole-filling" strategy.
    
    # We will try to remove each of the 50 terms one by one.
    # To save time, we will only try a few iterations per removal.
    # If we find a valid Rank 49, we return it.
    
    # Pre-compute the schoolbook matrices as lists for easy copying
    U_list = U_school.copy()
    V_list = V_school.copy()
    W_list = W_school.copy()
    
    # We'll try removing terms in a random order to avoid bias, or just sequential.
    # Sequential is deterministic and fine.
    
    # We limit the number of attempts to ensure we finish in time.
    # 50 attempts * (time per optimization)
    # If each optimization takes ~0.5s, 50 * 0.5 = 25s. This is safe.
    
    for remove_idx in range(rank_school):
        if time.time() > deadline - 5:
            break
            
        # Create initial guess for Rank 49
        # We take all columns except 'remove_idx'
        keep_indices = [i for i in range(rank_school) if i != remove_idx]
        
        U_init = U_school[:, keep_indices]
        V_init = V_school[:, keep_indices]
        W_init = W_school[:, keep_indices]
        
        # Add a small random perturbation to help escape local minima
        # The removed term needs to be "absorbed" by the others.
        # Without perturbation, the optimizer might just stay at 0 error for the existing terms 
        # and fail to cover the missing one, or converge to a local minimum that doesn't cover it.
        # However, the loss function will be high initially because the missing term is not covered.
        # The optimizer will try to adjust the other terms to cover it.
        
        # Let's add a small random noise to the initialization to encourage exploration
        noise_scale = 0.1
        U_init = U_init + np.random.randn(*U_init.shape) * noise_scale
        V_init = V_init + np.random.randn(*V_init.shape) * noise_scale
        W_init = W_init + np.random.randn(*W_init.shape) * noise_scale
        
        result = try_optimize(49, U_init, V_init, W_init, max_iter=20)
        if result:
            best_result = result
            best_rank = 49
            break
            
    return best_result
# EVOLVE-BLOCK-END
