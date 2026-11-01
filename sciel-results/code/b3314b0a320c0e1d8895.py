<<<<<<< SEARCH
def _search_rank_27(U28, V28, W28):
    """
    Search for a rank-27 decomposition by numerical optimization.
    Returns (U, V, W) with rank 27 if found, else None.
    """
    if time.time() > _DEADLINE - 30:
        return None
    
    nm, mp, pn = NM, MP, PN
    target_rank = 27
    
    # Initialize with the first 27 terms of the rank-28 decomposition
    U_init = np.array(U28[:, :target_rank], dtype=float)
    V_init = np.array(V28[:, :target_rank], dtype=float)
    W_init = np.array(W28[:, :target_rank], dtype=float)
    
    # Compute the target tensor
    T_target = np.zeros((nm, mp, pn))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                T_target[a, b, c] = sum(U28[a, r] * V28[b, r] * W28[c, r] for r in range(28))
    
    # Define the objective function
    def objective(params):
        U = params[:nm * target_rank].reshape(nm, target_rank)
        V = params[nm * target_rank:(nm + mp) * target_rank].reshape(mp, target_rank)
        W = params[(nm + mp) * target_rank:(nm + mp + pn) * target_rank].reshape(pn, target_rank)
        
        T = np.zeros((nm, mp, pn))
        for r in range(target_rank):
            T += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        
        diff = T - T_target
        return np.sum(diff ** 2)
    
    # Use scipy.optimize to minimize
    from scipy.optimize import minimize
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    
    if result.fun < 1e-10:
        U_opt = result.x[:nm * target_rank].reshape(nm, target_rank)
        V_opt = result.x[nm * target_rank:(nm + mp) * target_rank].reshape(mp, target_rank)
        W_opt = result.x[(nm + mp) * target_rank:(nm + mp + pn) * target_rank].reshape(pn, target_rank)
        
        # Round to nearest half-integer
        U_rounded = np.round(U_opt * 2) / 2
        V_rounded = np.round(V_opt * 2) / 2
        W_rounded = np.round(W_opt * 2) / 2
        
        # Verify
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None
=======
def _search_rank_27(U28, V28, W28):
    """
    Search for a rank-27 decomposition by numerical optimization.
    Returns (U, V, W) with rank 27 if found, else None.
    """
    if time.time() > _DEADLINE - 30:
        return None
    
    nm, mp, pn = NM, MP, PN
    target_rank = 27
    
    # Initialize with the first 27 terms of the rank-28 decomposition
    U_init = np.array(U28[:, :target_rank], dtype=float)
    V_init = np.array(V28[:, :target_rank], dtype=float)
    W_init = np.array(W28[:, :target_rank], dtype=float)
    
    # Compute the target tensor
    T_target = np.zeros((nm, mp, pn))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                T_target[a, b, c] = sum(U28[a, r] * V28[b, r] * W28[c, r] for r in range(28))
    
    # Define the objective function
    def objective(params):
        U = params[:nm * target_rank].reshape(nm, target_rank)
        V = params[nm * target_rank:(nm + mp) * target_rank].reshape(mp, target_rank)
        W = params[(nm + mp) * target_rank:(nm + mp + pn) * target_rank].reshape(pn, target_rank)
        
        T = np.zeros((nm, mp, pn))
        for r in range(target_rank):
            T += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        
        diff = T - T_target
        return np.sum(diff ** 2)
    
    # Use scipy.optimize to minimize
    from scipy.optimize import minimize
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14})
    
    if result.fun < 1e-8:
        U_opt = result.x[:nm * target_rank].reshape(nm, target_rank)
        V_opt = result.x[nm * target_rank:(nm + mp) * target_rank].reshape(mp, target_rank)
        W_opt = result.x[(nm + mp) * target_rank:(nm + mp + pn) * target_rank].reshape(pn, target_rank)
        
        # Try rounding to nearest half-integer
        U_rounded = np.round(U_opt * 2) / 2
        V_rounded = np.round(V_opt * 2) / 2
        W_rounded = np.round(W_opt * 2) / 2
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
        
        # Try rounding to nearest integer
        U_rounded = np.round(U_opt)
        V_rounded = np.round(V_opt)
        W_rounded = np.round(W_opt)
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None
>>>>>>> REPLACE
