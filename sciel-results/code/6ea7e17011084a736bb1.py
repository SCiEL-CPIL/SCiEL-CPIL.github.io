    # ... (previous code) ...
    
    best_params = res.x
    verts_final = get_inner_verts(best_params)
    
    # Compute true max L for the final answer
    x = verts_final[:, :, 0]
    y = verts_final[:, :, 1]
    
    term1 = x + y / np.sqrt(3)
    term2 = 2 * y / np.sqrt(3)
    term3 = -x + y / np.sqrt(3)
    term4 = -x - y / np.sqrt(3)
    term5 = -2 * y / np.sqrt(3)
    term6 = x - y / np.sqrt(3)
    
    all_terms = np.stack([term1, term2, term3, term4, term5, term6], axis=2)
    true_L = np.max(all_terms)
    
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang_deg = np.rad2deg(best_params[i*3+2])
        inner.append([cx, cy, ang_deg])
        
    return inner, (0.0, 0.0), true_L, 0.0
