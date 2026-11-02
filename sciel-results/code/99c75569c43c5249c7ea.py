def _polish(h, max_iters=500, initial_step=0.1):
    """
    Refine h using subgradient descent on the exact max overlap objective.
    """
    h = h.copy()
    best_h = h.copy()
    best_score = _compute_overlaps(h).max()
    step = initial_step
    
    for _ in range(max_iters):
        overlaps = _compute_overlaps(h)
        current_max = overlaps.max()
        
        # Subgradient of max is average of gradients of active sets
        # Active set: overlaps within eps of max
        eps = 1e-5
        active = overlaps >= (current_max - eps)
        
        if not np.any(active):
             # Fallback to single max
             active = np.zeros(len(overlaps), dtype=bool)
             active[np.argmax(overlaps)] = True
        
        # Compute average gradient
        # d/dh max(O) = (1/|A|) sum_{k in A} d O_k / d h
        # d O_k / d h = -(roll(h, -k) + roll(h, k))
        
        grad = np.zeros(N)
        active_indices = np.where(active)[0]
        for k_idx in active_indices:
            k = k_idx + 1
            grad -= (np.roll(h, -k) + np.roll(h, k))
        grad /= len(active_indices)
        
        # Gradient step
        h_new = h - step * grad
        h_new = _project(h_new)
        
        new_score = _compute_overlaps(h_new).max()
        
        if new_score < best_score:
            best_h = h_new
            best_score = new_score
            h = h_new
            # Decay step slowly to ensure convergence
            step *= 0.95 
        else:
            step *= 0.5
            if step < 1e-7:
                break
                
    return best_h
