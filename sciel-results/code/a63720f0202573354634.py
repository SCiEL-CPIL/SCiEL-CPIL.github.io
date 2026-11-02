import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=200, tol=1e-10, seed=0, time_budget=None):
    """
    Alternating Least Squares to decompose T into U, V, W.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rng = np.random.RandomState(seed)
    a, b, c = T.shape
    
    # Initialize factors with better scaling
    scale = 1.0 / np.sqrt(rank)
    U = rng.randn(a, rank) * scale
    V = rng.randn(b, rank) * scale
    W = rng.randn(c, rank) * scale
    
    # Precompute flattened target views
    T_ac = T.reshape(a, b * c)
    T_bc = T.transpose(1, 0, 2).reshape(b, a * c)
    T_cc = T.transpose(2, 0, 1).reshape(c, a * b)
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
            
        # Update U: solve T_ac = U @ Khatri(V,W)
        Khatri_VW = np.einsum('br,cr->bcr', V, W).reshape(b * c, rank).T
        try:
            U_new = np.linalg.lstsq(Khatri_VW, T_ac.T, rcond=None)[0].T
        except:
            U_new = U
        
        # Update V: solve T_bc = V @ Khatri(U,W)
        Khatri_UW = np.einsum('ar,cr->acr', U_new, W).reshape(a * c, rank).T
        try:
            V_new = np.linalg.lstsq(Khatri_UW, T_bc.T, rcond=None)[0].T
        except:
            V_new = V
        
        # Update W: solve T_cc = W @ Khatri(U,V)
        Khatri_UV = np.einsum('ar,br->abr', U_new, V_new).reshape(a * b, rank).T
        try:
            W_new = np.linalg.lstsq(Khatri_UV, T_cc.T, rcond=None)[0].T
        except:
            W_new = W
        
        # Check convergence
        U_diff = np.linalg.norm(U_new - U)
        V_diff = np.linalg.norm(V_new - V)
        W_diff = np.linalg.norm(W_new - W)
        
        U, V, W = U_new, V_new, W_new
        
        if U_diff + V_diff + W_diff < tol:
            break
    
    # Compute residual
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
    
    return U, V, W, residual

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Deadline guard
    start_time = time.time()
    deadline = 135.0  # Leave 15s margin
    
    best_result = None
    best_rank = rank_schoolbook
    best_residual = float('inf')
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try ranks in increasing order
    # For 2x3x9, known constructions suggest rank around 30-36
    # Start from a more reasonable lower bound
    target_ranks = list(range(30, 45))
    
    for r in target_ranks:
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        found = False
        best_local_residual = float('inf')
        best_local_U = best_local_V = best_local_W = None
        
        # Dynamic restart budget: more restarts for smaller ranks (harder to find)
        # Allocate remaining time proportionally
        remaining_time = deadline - 10 - elapsed
        ranks_remaining = len(target_ranks) - target_ranks.index(r)
        time_per_rank = remaining_time / max(ranks_remaining, 1)
        
        # Estimate time per ALS run (rough: 0.5-2s depending on rank)
        est_time_per_run = 1.5
        max_restarts = max(3, min(15, int(time_per_rank / est_time_per_run)))
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 8:
                break
            
            # Use diverse seeds
            seed = attempt * 100 + r * 7
            
            try:
                U, V, W, residual = _als_decompose(
                    T, r, 
                    max_iter=150, 
                    tol=1e-12, 
                    seed=seed,
                    time_budget=deadline - 5
                )
                
                if residual < best_local_residual:
                    best_local_residual = residual
                    best_local_U = U
                    best_local_V = V
                    best_local_W = W
                    
                # If residual is very small, we have a good solution
                if residual < 1e-8:
                    found = True
                    break
                    
            except Exception as e:
                continue
        
        if found and best_local_residual < 1e-8:
            best_result = {
                "U": best_local_U.tolist(),
                "V": best_local_V.tolist(),
                "W": best_local_W.tolist()
            }
            best_rank = r
            best_residual = best_local_residual
            break
    
    # If no improvement found, return schoolbook
    if best_result is None:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    else:
        return best_result
