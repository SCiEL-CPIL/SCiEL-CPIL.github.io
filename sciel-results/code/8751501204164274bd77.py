<<<<<<< SEARCH
def _alt_minimize(U0, V0, W0, T, r_target, n_trials=3, max_iter=150):
    """
    Alternating minimization to find a rank-r_target decomposition.
    T shape: (a, b, c) where a=n*m, b=m*p, c=p*n
    U shape: (a, r), V shape: (b, r), W shape: (c, r)
    Identity: T[i, j, k] = sum_d U[i,d] * V[j,d] * W[k,d]
    """
    a, b, c = T.shape
    r = r_target
    
    best_result = None
    best_err = np.inf
    
    for trial in range(n_trials):
        if _time_left() < 5:
            break
        
        # Initialize
        if trial == 0:
            U_curr = U0[:, :r].copy() if U0.shape[1] >= r else np.zeros((a, r))
            V_curr = V0[:, :r].copy() if V0.shape[1] >= r else np.zeros((b, r))
            W_curr = W0[:, :r].copy() if W0.shape[1] >= r else np.zeros((c, r))
        else:
            rng = np.random.RandomState(trial * 137 + 42)
            if U0.shape[1] >= r:
                U_curr = U0[:, :r] + rng.randn(a, r) * 0.05
                V_curr = V0[:, :r] + rng.randn(b, r) * 0.05
                W_curr = W0[:, :r] + rng.randn(c, r) * 0.05
            else:
                U_curr = rng.randn(a, r) * 0.1
                V_curr = rng.randn(b, r) * 0.1
                W_curr = rng.randn(c, r) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if _time_left() < 3:
                break
            
            # Optimize W
            M = np.einsum('id,jd->ijd', U_curr, V_curr).reshape(a * b, r)
            T_rhs = T.transpose(2, 0, 1).reshape(c, a * b).T
            try:
                W_new = np.linalg.lstsq(M, T_rhs, rcond=None)[0].T
            except:
                W_new = W_curr
            
            # Optimize U
            V_W = np.einsum('jd,kd->jkd', V_curr, W_curr)
            M_U = np.tile(V_W.reshape(b * c, r), (a, 1, 1))
            T_lhs = T.reshape(a, b * c)
            
            U_new = np.zeros((a, r))
            for i in range(a):
                try:
                    U_new[i, :] = np.linalg.lstsq(M_U[i, :, :].T, T_lhs[i, :], rcond=None)[0]
                except:
                    U_new[i, :] = U_curr[i, :]
            
            # Optimize V
            U_W = np.einsum('id,kd->ikd', U_curr, W_curr)
            M_V = np.tile(U_W.reshape(a * c, r), (b, 1, 1))
            T_rhs2 = T.transpose(0, 2, 1).reshape(a * c, b).T
            
            V_new = np.zeros((b, r))
            for j in range(b):
                try:
                    V_new[j, :] = np.linalg.lstsq(M_V[j, :, :].T, T_rhs2[j, :], rcond=None)[0]
                except:
                    V_new[j, :] = V_curr[j, :]
            
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            # Compute error
            T_recon = np.einsum('id,jd,kd->ijk', U_curr, V_curr, W_curr)
            err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            
            if err < best_err:
                best_err = err
                best_result = (U_curr.copy(), V_curr.copy(), W_curr.copy())
            
            if err < 1e-10:
                break
            if abs(err - prev_err) < 1e-14:
                break
            prev_err = err
    
    if best_result is not None:
        U_b, V_b, W_b = best_result
        # Try rounding to half-integers
        U_r = _round_half(U_b)
        V_r = _round_half(V_b)
        W_r = _round_half(W_b)
        
        T_recon_r = np.einsum('id,jd,kd->ijk', U_r, V_r, W_r)
        if np.allclose(T_recon_r, T, atol=1e-6, rtol=1e-6):
            return U_r, V_r, W_r, best_err
        else:
            return U_b, V_b, W_b, best_err
    
    return None
=======
def _greedy_minimize(T, r_target, n_trials=5, max_iter=200):
    """
    Greedy randomized search for rank-r_target decomposition.
    T shape: (a, b, c)
    """
    a, b, c = T.shape
    r = r_target
    
    best_result = None
    best_err = np.inf
    
    for trial in range(n_trials):
        if _time_left() < 5:
            break
        
        rng = np.random.RandomState(trial * 137 + 42)
        
        # Initialize with random rank-1 tensors or zero
        U_list = []
        V_list = []
        W_list = []
        T_curr = T.copy()
        
        # We will construct the decomposition rank by rank
        # For each rank step, we find the best rank-1 approximation
        # T_curr ~ u v w^T
        # Then T_curr = T_curr - u v w^T
        
        for step in range(r):
            if _time_left() < 2:
                break
            
            # Random restarts for the greedy step
            found_best = False
            for attempt in range(10):
                if _time_left() < 1:
                    break
                
                # Randomly initialize u, v, w
                u = rng.randn(a)
                v = rng.randn(b)
                w = rng.randn(c)
                
                # Iterate to find best rank-1 approx of T_curr
                # u ~ T_curr * v * w (vectorized)
                # v ~ T_curr * u * w
                # w ~ T_curr * u * v
                
                for _ in range(20):
                    # Update u
                    # T_curr is (a, b, c)
                    # contract over b, c
                    # u_new[i] = sum_{j,k} T_curr[i,j,k] * v[j] * w[k]
                    u = np.einsum('ijk,j,k->i', T_curr, v, w)
                    norm_u = np.linalg.norm(u)
                    if norm_u < 1e-10:
                        break
                    u /= norm_u
                    
                    # Update v
                    v = np.einsum('ijk,i,k->j', T_curr, u, w)
                    norm_v = np.linalg.norm(v)
                    if norm_v < 1e-10:
                        break
                    v /= norm_v
                    
                    # Update w
                    w = np.einsum('ijk,i,j->k', T_curr, u, v)
                    norm_w = np.linalg.norm(w)
                    if norm_w < 1e-10:
                        break
                    w /= norm_w
                
                # Check residual
                T_approx = np.einsum('i,j,k->ijk', u, v, w)
                T_new = T_curr - T_approx
                err = np.linalg.norm(T_new)
                
                if not found_best or err < best_err_step:
                    best_err_step = err
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                    found_best = True
            
            if not found_best:
                break
            
            # Commit the step
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            T_curr = T_curr - np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            
            # Early stop if residual is small
            if np.linalg.norm(T_curr) < 1e-9:
                break
        
        if len(U_list) == r:
            U_curr = np.column_stack(U_list)
            V_curr = np.column_stack(V_list)
            W_curr = np.column_stack(W_list)
            
            T_recon = np.einsum('id,jd,kd->ijk', U_curr, V_curr, W_curr)
            err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            
            if err < best_err:
                best_err = err
                best_result = (U_curr.copy(), V_curr.copy(), W_curr.copy())
                
                # Try rounding
                U_r = _round_half(U_curr)
                V_r = _round_half(V_curr)
                W_r = _round_half(W_curr)
                
                T_recon_r = np.einsum('id,jd,kd->ijk', U_r, V_r, W_r)
                if np.allclose(T_recon_r, T, atol=1e-6, rtol=1e-6):
                    return U_r, V_r, W_r, err
    
    if best_result is not None:
        return best_result
    
    return None
>>>>>>> REPLACE

