import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # 150s kill limit. Use 140s to be safe.
    deadline = start_time + 140.0
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Pre-compute flattened views of T for ALS
    T_flat_u = T.reshape(n * m, -1) # (nm, mp*pn)
    T_flat_v = T.reshape(m * p, -1) # (mp, nm*pn)
    T_flat_w = T.reshape(p * n, -1) # (pn, nm*mp)
    
    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Lower bound: max(n*m, m*p, p*n) = 16
    lower_bound = max(n * m, m * p, p * n)
    
    # We search from lower_bound up to rank_school - 1
    # If we find a valid decomposition at rank R, we return it.
    # To maximize chances, we should allocate more time to lower ranks?
    # Or, since lower ranks are harder (more constraints), we might fail quickly.
    # Higher ranks (close to 48) are easier to find but we want MINIMAL.
    # So we MUST check 16, 17, 18... in order.
    
    # Optimization: 
    # For each rank R, run multiple trials.
    # If R is close to lower_bound, use more random restarts.
    # If R is close to rank_school, use fewer (likely to succeed quickly if it exists).
    
    # Time budget allocation:
    # Total time ~ 140s.
    # Ranges: 16 to 47 (32 ranks).
    # If we assume uniform distribution of difficulty, we can spend ~4s per rank.
    # But lower ranks are harder.
    
    # Strategy:
    # 1. Try R from 16 to 47.
    # 2. For each R, try several initializations.
    # 3. Initializations:
    #    a. SVD-based (Top R singular vectors)
    #    b. Random Gaussian
    #    c. Random Integer {-1, 0, 1} (Scaled)
    #    d. Random Half-Integer {-0.5, 0, 0.5, 1.0, 1.5}
    
    def run_als(U, V, W, R, max_iter=200):
        """
        Runs ALS optimization for U, V, W.
        Returns (U, V, W, error).
        """
        current_error = np.inf
        prev_error = np.inf
        
        for it in range(max_iter):
            # Check time
            if time.time() > deadline - 2:
                break
            
            # Optimize U
            # B_u[r] = (V[:, r] \otimes W[:, r]).reshape(-1)
            # Efficient construction:
            # V[:, r] is (mp, 1), W[:, r] is (pn, 1)
            # Outer product is (mp, pn). Flatten to (mp*pn, 1).
            # We need a matrix B_u of shape (R, mp*pn) where B_u[r, :] is the flattened outer product.
            # Then U = argmin || T_flat_u - B_u.T @ U ||
            
            # Vectorized outer product construction
            # V: (mp, R), W: (pn, R)
            # We want B_u: (R, mp*pn)
            # B_u[r, j*pn + k] = V[j, r] * W[k, r]
            
            # Using broadcasting:
            # V_T: (R, mp), W_T: (R, pn)
            # Outer: (R, mp, pn) -> reshape (R, mp*pn)
            V_T = V.T # (R, mp)
            W_T = W.T # (R, pn)
            B_u = np.einsum('ir,is->irs', V_T, W_T).reshape(R, -1)
            
            # Solve U = (B_u B_u^T)^-1 B_u T_flat_u^T ?
            # Or np.linalg.lstsq(B_u.T, T_flat_u)
            # T_flat_u is (nm, mp*pn). We want U (nm, R).
            # Equation: T_flat_u \approx B_u.T @ U
            # So U = lstsq(B_u.T, T_flat_u)
            
            try:
                U, _, _, _ = np.linalg.lstsq(B_u.T, T_flat_u, rcond=None)
            except:
                pass
                
            # Optimize V
            # B_v[r, i*pn + k] = U[i, r] * W[k, r]
            # U: (nm, R), W: (pn, R)
            U_T = U.T # (R, nm)
            W_T = W.T # (R, pn)
            B_v = np.einsum('ir,is->irs', U_T, W_T).reshape(R, -1)
            
            try:
                V, _, _, _ = np.linalg.lstsq(B_v.T, T_flat_v, rcond=None)
            except:
                pass
                
            # Optimize W
            # B_w[r, i*mp + j] = U[i, r] * V[j, r]
            # U: (nm, R), V: (mp, R)
            U_T = U.T # (R, nm)
            V_T = V.T # (R, mp)
            B_w = np.einsum('ir,is->irs', U_T, V_T).reshape(R, -1)
            
            try:
                W, _, _, _ = np.linalg.lstsq(B_w.T, T_flat_w, rcond=None)
            except:
                pass
                
            # Compute Error
            # recon = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
            # T - recon
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            current_error = np.linalg.norm(T - recon)
            
            # Convergence
            if abs(prev_error - current_error) < 1e-12:
                break
            prev_error = current_error
            
        return U, V, W, current_error

    def try_decomposition(R, trial_seed):
        """
        Attempts to find a decomposition of rank R.
        Returns (U, V, W) if successful, else None.
        """
        np.random.seed(trial_seed)
        
        # Initialization strategies
        # 1. SVD
        if trial_seed % 3 == 0:
            U_init, _, _ = np.linalg.svd(T_flat_u, full_matrices=False)
            U_init = U_init[:, :R]
            V_init, _, _ = np.linalg.svd(T_flat_v, full_matrices=False)
            V_init = V_init[:, :R]
            W_init, _, _ = np.linalg.svd(T_flat_w, full_matrices=False)
            W_init = W_init[:, :R]
        elif trial_seed % 3 == 1:
            # Random Gaussian
            scale = 1.0
            U_init = np.random.randn(n * m, R) * scale
            V_init = np.random.randn(m * p, R) * scale
            W_init = np.random.randn(p * n, R) * scale
        else:
            # Random Integer {-1, 0, 1}
            U_init = np.random.randint(-1, 2, (n * m, R)).astype(float)
            V_init = np.random.randint(-1, 2, (m * p, R)).astype(float)
            W_init = np.random.randint(-1, 2, (p * n, R)).astype(float)
            # Scale to match norm roughly?
            # The tensor norm is sqrt(48).
            # If U,V,W are small integers, the product might be small.
            # Let's scale by 1.0 for now.
            
        U, V, W, error = run_als(U_init, V_init, W_init, R, max_iter=100)
        
        if error < 1e-8:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            recon_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            error_rounded = np.linalg.norm(T - recon_rounded)
            
            if error_rounded < 1e-6:
                return U_rounded, V_rounded, W_rounded
        return None

    # Search Loop
    for R in range(lower_bound, rank_school):
        if time.time() > deadline - 10:
            break
            
        # Determine number of trials based on remaining time and rank
        # Lower ranks need more trials.
        if R < 20:
            max_trials = 20
        elif R < 30:
            max_trials = 10
        else:
            max_trials = 5
            
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
                
            seed = 42 + R * 1000 + trial
            result = try_decomposition(R, seed)
            if result is not None:
                U_rounded, V_rounded, W_rounded = result
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = R
                found = True
                break
                
        if found:
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
