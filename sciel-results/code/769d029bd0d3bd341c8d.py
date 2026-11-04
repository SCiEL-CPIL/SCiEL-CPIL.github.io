import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_indices = n * m * p
    
    def residual_func(params, r):
        # params shape: (size_u + size_v + size_w) * r
        # Reshape to U (size_u, r), V (size_v, r), W (size_w, r)
        u_flat = params[:size_u * r].reshape(size_u, r)
        v_flat = params[size_u * r : size_u * r + size_v * r].reshape(size_v, r)
        w_flat = params[size_u * r + size_v * r :].reshape(size_w, r)
        
        # Compute sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We need to compute this for all i, j, k.
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U = u_flat.reshape(n, m, r)
        V = v_flat.reshape(m, p, r)
        W = w_flat.reshape(p, n, r)
        
        # We want T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # Note: W is indexed [k, n, i] in the reshape? 
        # w_flat index is k*n + i. So W[k, i, c].
        # So W reshaped is (p, n, r).
        
        # We need to sum over c (axis 2).
        # U[i, j, c] -> shape (n, m, r)
        # V[j, k, c] -> shape (m, p, r) -> need to align to (n, m, p, r)?
        # W[k, i, c] -> shape (p, n, r) -> need to align.
        
        # Let's use einsum or broadcasting.
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        
        # Term U[i, j, c] V[j, k, c] W[k, i, c]
        # Sum over c.
        
        # We can do this by iterating c? r is small (<=30).
        # Or use einsum: 'ijc,jkc,kic->ijk'
        # But W is (p, n, r). The index order in einsum for W would be 'kic' if we treat the first axis as k, second as i.
        # W reshaped is (p, n, r). So axis 0 is k, axis 1 is i, axis 2 is c.
        # So W[k, i, c].
        
        try:
            # einsum string:
            # U: i j c
            # V: j k c
            # W: k i c
            # Result: i j k
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W, optimize=True)
        except Exception:
            # Fallback to loop if einsum fails or is too slow
            T = np.zeros((n, m, p))
            for c in range(r):
                T += U[:, :, c][:, :, np.newaxis] * V[:, c, :][np.newaxis, :, :] * W[c, :, :].T[:, np.newaxis, :]
                # W[c, :, :] is (p, n). We need W[k, i].
                # W[c, k, i] in the loop? No, W is (p, n, r).
                # W[:, :, c] is (p, n).
                # We need W[k, i].
                # So W_part = W[:, :, c].T? No.
                # W[:, :, c] has indices [k, i].
                # So W_part = W[:, :, c].
                # We need to broadcast (n, m, 1) * (1, m, p) * (n, 1, p)?
                # U_part: (n, m, 1)
                # V_part: (1, m, p)
                # W_part: (n, 1, p) ?
                # W[k, i] needs to be at [i, 0, k].
                # W[:, :, c] is (p, n).
                # W[:, :, c].T is (n, p).
                # W.T[i, k] = W[k, i].
                # So W_part = W[:, :, c].T[:, np.newaxis, :] -> (n, 1, p).
                # T += U[:, :, c][:, :, np.newaxis] * V[:, c, :][np.newaxis, :, :] * W[:, :, c].T[:, np.newaxis, :]
                # Let's re-verify V.
                # V is (m, p, r). V[:, c, :] is (m, p).
                # We need V[j, k].
                # V_part: (1, m, p).
                # V[:, c, :][np.newaxis, :, :] is (1, m, p). Correct.
                # U[:, :, c] is (n, m).
                # U_part: (n, m, 1).
                # U[:, :, c][:, :, np.newaxis] is (n, m, 1). Correct.
                # W[:, :, c] is (p, n).
                # W.T is (n, p).
                # W_part: (n, 1, p).
                # W[:, :, c].T[:, np.newaxis, :] is (n, 1, p). Correct.
                
        # Residuals: T - 1
        return (T - 1).flatten()

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    fallback = schoolbook()
    best_result = fallback
    best_rank = n * m * p
    
    # Search for lower ranks
    # Start from a reasonable lower bound.
    # R(2,3,5) is known to be <= 29.
    # Let's try 29 down to 20.
    for r in range(29, 19, -1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        total_vars = (size_u + size_v + size_w) * r
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale small to avoid overflow
            params = np.random.randn(total_vars) * 0.5
            
            try:
                # Use least_squares
                # max_nfev is number of function evaluations
                result = opt.least_squares(
                    residual_func, 
                    params, 
                    args=(r,), 
                    method='trf',
                    max_nfev=1000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
            except Exception:
                continue
                
            if result.success or result.cost < 1e-10:
                # Check residual
                if result.cost < 1e-8:
                    # Extract solution
                    u_flat = result.x[:size_u * r].reshape(size_u, r)
                    v_flat = result.x[size_u * r : size_u * r + size_v * r].reshape(size_v, r)
                    w_flat = result.x[size_u * r + size_v * r :].reshape(size_w, r)
                    
                    # Attempt to snap to half-integers
                    U_snap = np.round(u_flat * 2) / 2
                    V_snap = np.round(v_flat * 2) / 2
                    W_snap = np.round(w_flat * 2) / 2
                    
                    # Verify snapped solution
                    # Construct flat params for snapped
                    params_snap = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                    res_snap = residual_func(params_snap, r)
                    err_snap = np.sum(res_snap**2)
                    
                    if err_snap < 1e-8:
                        # Use snapped solution
                        U_sol = U_snap
                        V_sol = V_snap
                        W_sol = W_snap
                    else:
                        # Use float solution if it's good enough
                        # Check float error again to be sure
                        res_float = residual_func(result.x, r)
                        err_float = np.sum(res_float**2)
                        if err_float < 1e-8:
                            U_sol = u_flat
                            V_sol = v_flat
                            W_sol = w_flat
                        else:
                            continue # Not good enough
                    
                    # Convert to list of lists
                    U_list = U_sol.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_sol.tolist()
                    
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    break # Found solution for this rank, try lower rank
                    
    return best_result
