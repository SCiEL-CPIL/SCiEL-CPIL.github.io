import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target Tensor T
    # T[u, v, w] where u = i*m+j, v = j*p+k, w = k*n+i
    # T is 1 if u, v, w correspond to a valid (i,j,k) term
    # i in [0,n), j in [0,m), k in [0,p)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    T_flat = T.flatten()
    
    # Helper to check validity
    def check_decomposition(U, V, W):
        # Compute T_hat
        # U: (dim_u, R), V: (dim_v, R), W: (dim_w, R)
        # T_hat[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
        # Use einsum
        T_hat = np.einsum('uc,vc,wc->uvw', U, V, W)
        diff = T_hat - T
        max_diff = np.max(np.abs(diff))
        return max_diff < 1e-6, T_hat

    # Helper to snap to half-integers
    def snap_to_half_int(mat):
        return np.round(mat * 2) / 2.0

    # Try to find a decomposition of rank R
    def find_decomposition(R, max_restarts=10, time_limit=10):
        import scipy.optimize as opt
        
        best_err = float('inf')
        best_params = None
        
        start_time = time.time()
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            # Scale: entries are around 1.
            # Use normal distribution
            U_init = np.random.randn(dim_u, R)
            V_init = np.random.randn(dim_v, R)
            W_init = np.random.randn(dim_w, R)
            
            # Normalize columns? Not strictly necessary but helps.
            # Let's just use random.
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss(params):
                U = params[:dim_u*R].reshape(dim_u, R)
                V = params[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                W = params[dim_u*R+dim_v*R:].reshape(dim_w, R)
                
                T_hat = np.einsum('uc,vc,wc->uvw', U, V, W)
                return (T_hat - T).flatten()
            
            try:
                res = opt.least_squares(loss, params, method='lm', max_nfev=100)
                if res.cost < best_err:
                    best_err = res.cost
                    best_params = res.x
            except Exception:
                pass
        
        if best_params is None:
            return None
            
        U = best_params[:dim_u*R].reshape(dim_u, R)
        V = best_params[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
        W = best_params[dim_u*R+dim_v*R:].reshape(dim_w, R)
        
        # Snap to half-integers
        U_snap = snap_to_half_int(U)
        V_snap = snap_to_half_int(V)
        W_snap = snap_to_half_int(W)
        
        # Check snapped solution
        valid, _ = check_decomposition(U_snap, V_snap, W_snap)
        if valid:
            return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        # Check original float solution (just in case, though task requires int/half-int)
        # The task says "integer/half-integer entries".
        # So we MUST return snapped values if they are valid.
        # If snapped is not valid, this rank failed.
        
        return None

    # Try ranks from 35 down to 30
    # Schoolbook is 36.
    # We want to beat 36.
    
    # Deadline guard
    global_start = time.time()
    deadline = global_start + 140 # 10s margin
    
    best_result = None
    
    # Try R = 35, 34, 33, 32, 31, 30
    for R in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
            
        # Allocate more time for lower ranks?
        # Lower ranks are harder to find, but if they exist, they are better.
        # Let's give 20s to each rank.
        result = find_decomposition(R, max_restarts=50, time_limit=20)
        
        if result is not None:
            best_result = result
            break # Found a valid one, stop (lowest R tried first? No, we try high to low? 
                  # Wait, we want MINIMAL rank.
                  # So we should try LOW to HIGH?
                  # No, if we find rank 30, we don't care about 35.
                  # So we should try 30, 31, 32...
                  # But 30 might not exist.
                  # Let's try 30 first. If found, return.
                  # If not, try 31, etc.
    
    # If the loop above was 30->35, it would stop at first success.
    # Let's re-structure:
    # Try 30, 31, 32, 33, 34, 35.
    
    # Re-do logic:
    # The previous loop was 35 down to 30.
    # If I find 35, I stop. But 30 is better.
    # So I MUST iterate 30, 31, 32...
    
    # Let's restart the search logic in the code below.
    
    if best_result is None:
        for R in [30, 31, 32, 33, 34, 35]:
            if time.time() > deadline:
                break
            result = find_decomposition(R, max_restarts=50, time_limit=20)
            if result is not None:
                best_result = result
                break

    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert numpy arrays to lists of lists for the return interface
    # The seed returns lists of lists.
    return {
        "U": best_result["U"].tolist(),
        "V": best_result["V"].tolist(),
        "W": best_result["W"].tolist()
    }
