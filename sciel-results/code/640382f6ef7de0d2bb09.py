def _try_rank_13(n, m, p, deadline):
    """Try to find a rank-13 decomposition using Alternating Least Squares (ALS)."""
    import numpy as np
    import time
    
    start_time = time.time()
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    # Build the target tensor T
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Get the rank-14 solution as a starting point
    rank_14 = _construct_rank_14(n, m, p)
    U14 = np.array(rank_14['U'], dtype=float)
    V14 = np.array(rank_14['V'], dtype=float)
    W14 = np.array(rank_14['W'], dtype=float)
    
    best_error = float('inf')
    best_solution = None
    
    # Strategy: Drop one of the 14 terms and try to fit the remaining 13.
    # We'll try dropping each term and running ALS.
    
    for drop_idx in range(14):
        if time.time() - start_time > deadline - 10:
            break
            
        # Initialize U, V, W by taking the 13 terms excluding drop_idx
        U_init = np.delete(U14, drop_idx, axis=1)
        V_init = np.delete(V14, drop_idx, axis=1)
        W_init = np.delete(W14, drop_idx, axis=1)
        
        # Run ALS
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for iter in range(100): # Limit iterations per drop
            if time.time() - start_time > deadline - 10:
                break
                
            # Solve for U: T = sum_r U_r * (V_r * W_r)
            # Reshape T to (u_dim, v_dim * w_dim)
            T_u = T.reshape(u_dim, v_dim * w_dim)
            # Build matrix for U: each column is V_r * W_r (flattened)
            M_u = np.zeros((v_dim * w_dim, 13))
            for r in range(13):
                M_u[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            # U = T_u * M_u * inv(M_u^T M_u)
            try:
                U = np.linalg.lstsq(M_u, T_u, rcond=None)[0].T
            except:
                break
                
            # Solve for V: T = sum_r (U_r * W_r) * V_r
            T_v = T.transpose(1, 0, 2).reshape(v_dim, u_dim * w_dim)
            M_v = np.zeros((u_dim * w_dim, 13))
            for r in range(13):
                M_v[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            try:
                V = np.linalg.lstsq(M_v, T_v, rcond=None)[0].T
            except:
                break
                
            # Solve for W: T = sum_r (U_r * V_r) * W_r
            T_w = T.transpose(2, 0, 1).reshape(w_dim, u_dim * v_dim)
            M_w = np.zeros((u_dim * v_dim, 13))
            for r in range(13):
                M_w[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            try:
                W = np.linalg.lstsq(M_w, T_w, rcond=None)[0].T
            except:
                break
                
            # Calculate error
            T_approx = np.zeros_like(T)
            for r in range(13):
                T_approx += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            error = np.linalg.norm(T - T_approx)
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
                
            if error < 1e-6:
                break
                
        if best_error < 1e-6:
            break

    if best_solution is not None and best_error < 1e-6:
        U, V, W = best_solution
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    return None
