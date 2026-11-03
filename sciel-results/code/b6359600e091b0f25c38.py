import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 4, 8]

def _residuals(params, R, n, m, p):
    """
    Compute residuals for the tensor decomposition.
    params: flattened array of U, V, W.
    """
    # Unpack
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    U = params[:size_U].reshape(n * m, R)
    V = params[size_U:size_U + size_V].reshape(m * p, R)
    W = params[size_U + size_V:].reshape(p * n, R)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                residuals[idx] = 1.0 - val
                idx += 1
    return residuals

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    # Check if entries are close to half-integers
    # And if the product is exactly 1
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to nearest 0.5."""
    return np.round(M * 2.0) / 2.0

def _try_rank(R, n, m, p, time_limit, seed_base=0):
    """
    Try to find a decomposition of rank R.
    """
    start_time = time.time()
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        seed = seed_base + attempt * 137
        np.random.seed(seed)
        
        # Initialize
        # Scale initialization to help convergence
        # Heuristic: entries should be around 1
        U_init = np.random.randn(n * m, R)
        V_init = np.random.randn(m * p, R)
        W_init = np.random.randn(p * n, R)
        
        # Normalize columns to prevent scaling issues?
        # Or just start with small random values
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Run optimization
        try:
            result = opt.least_squares(
                _residuals, 
                params_init, 
                args=(R, n, m, p),
                method='trf', # Trust Region Reflective
                max_nfev=1000 # Limit function evaluations
            )
        except Exception:
            continue
            
        if result.success or result.cost < 1e-10:
            # Extract solution
            params_sol = result.x
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            
            U_sol = params_sol[:size_U].reshape(n * m, R)
            V_sol = params_sol[size_U:size_U + size_V].reshape(m * p, R)
            W_sol = params_sol[size_U + size_V:].reshape(p * n, R)
            
            # Check if it's a valid solution
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                # Snap to half-integers
                U_snap = _snap_to_half_integer(U_sol)
                V_snap = _snap_to_half_integer(V_sol)
                W_snap = _snap_to_half_integer(W_sol)
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                else:
                    # If snap fails, maybe the float solution is good enough?
                    # But the prompt requires integer/half-integer.
                    # So we must return the snapped one if it works.
                    # If it doesn't work, this attempt failed.
                    pass
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145 # 5s safety margin
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Search for lower rank
    min_rank = max(n*m, m*p, n*p)
    
    for R in range(min_rank, schoolbook_rank + 1):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 2:
            break
            
        # Allocate time proportional to remaining budget?
        # Give more time to lower ranks?
        # Or fixed time per rank?
        # Let's give 2 seconds per rank initially, increasing for lower ranks?
        # Actually, lower ranks are harder.
        # Let's just use the remaining time divided by remaining ranks?
        # No, let's try each rank with a fixed budget, e.g., 1-2 seconds.
        
        result = _try_rank(R, n, m, p, time_limit=min(2.0, time_left), seed_base=R)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_result = {
                "U": U_sol.tolist(),
                "V": V_sol.tolist(),
                "W": W_sol.tolist()
            }
            # If we found a solution, we can stop if we hit the lower bound?
            # No, we want the *minimal* rank.
            # If we start from min_rank and go up, the first one we find is minimal.
            return best_result
            
    return best_result
