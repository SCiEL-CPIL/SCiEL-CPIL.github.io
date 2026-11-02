def _balance_factors(U, V, W):
    U_norms = np.linalg.norm(U, axis=0)
    V_norms = np.linalg.norm(V, axis=0)
    W_norms = np.linalg.norm(W, axis=0)
    
    # Avoid zero
    U_norms[U_norms < 1e-9] = 1e-9
    V_norms[V_norms < 1e-9] = 1e-9
    W_norms[W_norms < 1e-9] = 1e-9
    
    scales = (U_norms * V_norms * W_norms) ** (1.0/3.0)
    
    U = U / scales
    V = V / scales
    W = W / scales
    
    return U, V, W
