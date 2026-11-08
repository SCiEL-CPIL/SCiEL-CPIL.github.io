import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor (sparse)
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                
    # Schoolbook solution as fallback
    best_rank = n * m * p
    U_sb = np.zeros((dim_u, best_rank))
    V_sb = np.zeros((dim_v, best_rank))
    W_sb = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    # Strategy: Try to reduce rank by merging pairs of terms
    # For each pair of schoolbook terms, check if they can be merged into one term
    # A merge is possible if the outer products are linearly dependent in a certain way
    
    # More practical: use optimization starting from perturbed schoolbook
    
    # Try ranks from best_rank - 1 down, but with smarter initialization
    # Key insight: start from schoolbook and try to make one column zero (rank reduction)
    
    for R in range(best_rank - 1, 34, -1):
        if time.time() - start_time > time_limit:
            break
        
        # For this rank, try multiple strategies
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            seed_val = 42 + R * 1000 + restart * 7
            np.random.seed(seed_val)
            
            # Strategy 1: Start from schoolbook and randomly perturb, then optimize
            # Take R+1 columns from schoolbook, add random noise, optimize to rank R
            # Actually, we need to construct a rank-R decomposition directly
            
            # Better: random initialization but with structure
            # Initialize U, V, W with small random values
            scale = 0.5
            U_init = np.random.randn(dim_u, R) * scale
            V_init = np.random.randn(dim_v, R) * scale
            W_init = np.random.randn(dim_w, R) * scale
            
            # Alternative: initialize with a subset of schoolbook terms + random
            if restart < 2:
                # Use a structured initialization
                # Pick R random schoolbook terms
                indices = np.random.choice(best_rank, R, replace=False)
                U_init = U_sb[:, indices].copy()
                V_init = V_sb[:, indices].copy()
                W_init = W_sb[:, indices].copy()
                # Add small random perturbation
                U_init += np.random.randn(dim_u, R) * 0.1
                V_init += np.random.randn(dim_v, R) * 0.1
                W_init += np.random.randn(dim_w, R) * 0.1
            
            def res_func(params, R=R):
                U = params[:dim_u*R].reshape(dim_u, R)
                V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                return (T_recon - target).flatten()
                
            try:
                params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                result = least_squares(
                    res_func, 
                    params_init, 
                    method='lm', 
                    max_nfev=300,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                # Check if the cost is very low
                if result.cost < 1e-6:
                    params = result.x
                    U = params[:dim_u*R].reshape(dim_u, R)
                    V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                    W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify
                    T_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, target, atol=1e-9):
                        best_rank = R
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        break # Found a valid solution for this R, move to next lower R
                        
                # Also try with the raw (unsnaped) solution if close enough
                if result.cost < 1e-8:
                    params = result.x
                    U = params[:dim_u*R].reshape(dim_u, R)
                    V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                    W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                    
                    T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
                    if np.allclose(T_check, target, atol=1e-8):
                        # Try to find a cleaner representation
                        # Snap to nearest 0.25
                        U_snap2 = np.round(U * 4) / 4
                        V_snap2 = np.round(V * 4) / 4
                        W_snap2 = np.round(W * 4) / 4
                        T_snap2 = np.einsum('ir,jr,kr->ijk', U_snap2, V_snap2, W_snap2)
                        if np.allclose(T_snap2, target, atol=1e-8):
                            best_rank = R
                            best_U = U_snap2
                            best_V = V_snap2
                            best_W = W_snap2
                            break
            except Exception:
                continue
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
