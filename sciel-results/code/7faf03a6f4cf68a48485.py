<<<<<<< SEARCH
    w, h, r, xs, ys = best_config
    
    # Final safety adjustment
    r_final = r * (1 - 1e-5)
    
    return np.column_stack([xs, ys, np.full(N, r_final)])
=======
    w, h, r, xs, ys = best_config
    
    # Final safety adjustment - shrink slightly to ensure validity
    r_final = r * (1 - 1e-4)
    
    # Scale positions to match the shrunk radius (keep centers, just adjust radius)
    # Actually, we should just return with the shrunk radius
    return np.column_stack([xs, ys, np.full(N, r_final)])
>>>>>>> REPLACE
