<<<<<<< SEARCH
def _random_rank_search(n, m, p, target_rank, n_trials=50, max_iter=200, seed=42):
    """
    Try to find a decomposition of given rank using random initialization + ALS-style refinement.
    Returns dict or None.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor (flattened)
    T_flat = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_flat[u * mp * pn + v * pn + w] = 1.0
    
    # Reshape to (nm, mp*pn) for ALS
    T_mat = T_flat.reshape(nm, mp * pn)
    
    best_result = None
    
    for trial in range(n_trials):
        if _remaining_time() < 10.0:
            break
            
        # Random initialization
        U = rng.randn(nm, target_rank) * 0.1
        V = rng.randn(mp, target_rank) * 0.1
        W = rng.randn(pn, target_rank) * 0.1
        
        # ALS iterations
        for it in range(max_iter):
            if _remaining_time() < 5.0:
                    break
            
            # Update U: for each u, U[u,:] = (V^T W^T T_u) / (V^T W^T V^T W)
            # T_u[v,w] = T[u, v, w]
            # sum_v,w T[u,v,w] V[v,r] W[w,r'] for all r,r'
            # = sum_r' W[:,r']^T * (T_u @ V[:,r'])
            
            # More efficient: 
            # For each u: U[u,:] = (sum_w T[u,:,w] W[w,:])^T @ V^T ... 
            # Actually: U[u,r] = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
            # Let's compute: M_u = sum_w T[u,:,w] * W[w,:]  -> (mp, target_rank)
            # Then U[u,:] = M_u @ V^T  -> no, that gives (mp, mp)...
            # U[u,r] = sum_v sum_w T[u,v,w] V[v,r] W[w,r]
            # = (T_u * (V @ W^T)^T).sum()  where T_u is (mp,pn)
            # = T_u.flatten() dot (V @ W^T).flatten()
            
            # For efficiency, precompute VW = V @ W^T -> (mp, pn)
            VW = V @ W.T  # (mp, pn)
            
            for u in range(nm):
                # U[u,r] = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
                # = T[u,:].flatten() dot (V @ W.T).flatten()  -- NO, that's not right
                # U[u,r] = sum_v sum_w T[u,v,w] V[v,r] W[w,r]
                # This is elementwise: T[u] * (V[:,r] W[:,r]) summed
                # = T[u].flatten() dot np.outer(V[:,r], W[:,r]).flatten() for each r
                # = T[u].flatten() dot (V * W).flatten() ... no, V and W have different shapes
                
                # Better: U[u,r] = (T[u] * (V[:,r][:,None] * W[:,r][None,:])).sum()
                # For all r: U[u,:] = T[u].flatten() dot (V * W.T).flatten() -- still not right
                
                # Let's just do it directly but vectorized:
                # U[u,:] = np.array([T[u].flatten() @ (V[:,r] * W[:,r]).flatten() for r in range(target_rank)])
                # = T[u].flatten() @ np.einsum('vr,wr->vwr', V, W).reshape(mp*pn)
                # Precompute: outer_VW = np.einsum('vr,wr->vwr', V, W) -> (mp, pn, target_rank)
                # Then U[u,:] = T[u].reshape(mp, pn) * outer_VW, summed over v,w
                
                # Even better: U[u,:] = np.einsum('vw,vwr->r', T[u], outer_VW)
                # But we need outer_VW for each trial. Let's precompute once per iteration.
                
                pass  # We'll handle this below with precomputed outer_VW
            
            # Precompute outer_VW = einsum('vr,wr->vwr', V, W) -> (mp, pn, target_rank)
            outer_VW = np.einsum('vr,wr->vwr', V, W)  # (mp, pn, R)
            
            for u in range(nm):
                Tu = T_mat[u].reshape(mp, pn)
                # U[u,r] = sum_{v,w} Tu[v,w] * outer_VW[v,w,r]
                U[u, :] = np.einsum('vw,vwr->r', Tu, outer_VW)
            
            # Update V: V[v,r] = sum_{u,w} T[u,v,w] U[u,r] W[w,r]
            # Precompute outer_UW = einsum('ur,wr->uwr', U, W) -> (nm, pn, R)
            outer_UW = np.einsum('ur,wr->uwr', U, W)  # (nm, pn, R)
            
            for v in range(mp):
                # T[:,v,:] is (nm, pn)
                Tv = T_flat.reshape(nm, mp, pn)[:, v, :]
                V[v, :] = np.einsum('uw,uwr->r', Tv, outer_UW)
            
            # Update W: W[w,r] = sum_{u,v} T[u,v,w] U[u,r] V[v,r]
            # Precompute outer_UV = einsum('ur,vr->uvr', U, V) -> (nm, mp, R)
            outer_UV = np.einsum('ur,vr->uvr', U, V)  # (nm, mp, R)
            
            for w in range(pn):
                Tw = T_flat.reshape(nm, mp, pn)[:, :, w]
                W[w, :] = np.einsum('uv,uvr->r', Tw, outer_UV)
        
        # Check if this decomposition is exact
        if _remaining_time() < 5.0:
            break
            
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try rounding to nearest 0.5
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        U_list = [[float(x) for x in row] for row in U_round]
        V_list = [[float(x) for x in row] for row in V_round]
        W_list = [[float(x) for x in row] for row in W_round]
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
=======
def _random_rank_search(n, m, p, target_rank, n_trials=30, max_iter=100, seed=42):
    """
    Try to find a decomposition of given rank using random initialization + ALS-style refinement.
    Returns dict or None.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor (flattened)
    T_flat = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_flat[u * mp * pn + v * pn + w] = 1.0
    
    T_3d = T_flat.reshape(nm, mp, pn)
    
    for trial in range(n_trials):
        if _remaining_time() < 10.0:
            break
            
        # Random sparse initialization
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for r in range(target_rank):
            n_nonzero_u = rng.randint(2, 4)
            n_nonzero_v = rng.randint(2, 4)
            n_nonzero_w = rng.randint(2, 4)
            
            u_indices = rng.choice(nm, n_nonzero_u, replace=False)
            v_indices = rng.choice(mp, n_nonzero_v, replace=False)
            w_indices = rng.choice(pn, n_nonzero_w, replace=False)
            
            for idx in u_indices:
                U[idx, r] = rng.choice([-1, 1])
            for idx in v_indices:
                V[idx, r] = rng.choice([-1, 1])
            for idx in w_indices:
                W[idx, r] = rng.choice([-1, 1])
        
        # ALS iterations
        for it in range(max_iter):
            if _remaining_time() < 5.0:
                break
            
            # Update U
            outer_VW = np.einsum('vr,wr->vwr', V, W)
            for u in range(nm):
                U[u, :] = np.einsum('vw,vwr->r', T_3d[u], outer_VW)
            
            # Update V
            outer_UW = np.einsum('ur,wr->uwr', U, W)
            for v in range(mp):
                V[v, :] = np.einsum('uw,uwr->r', T_3d[:, v, :], outer_UW)
            
            # Update W
            outer_UV = np.einsum('ur,vr->uvr', U, V)
            for w in range(pn):
                W[w, :] = np.einsum('uv,uvr->r', T_3d[:, :, w], outer_UV)
        
        if _remaining_time() < 5.0:
            break
            
        # Try rounding to nearest 0.5
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        U_list = [[float(x) for x in row] for row in U_round]
        V_list = [[float(x) for x in row] for row in V_round]
        W_list = [[float(x) for x in row] for row in W_round]
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
>>>>>>> REPLACE

