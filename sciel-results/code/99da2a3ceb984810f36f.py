<<<<<<< SEARCH
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _local_refine(best_pos, best_radii, deadline, lam=200.0)
    
    # Ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    return np.column_stack([best_pos, best_radii])
=======
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _local_refine(best_pos, best_radii, deadline, lam=200.0)
    
    # Final grow-and-push to squeeze out remaining space
    if time.time() < deadline - 2:
        try:
            gp_pos, gp_radii = _grow_and_push(best_pos, best_radii, deadline, max_iter=200)
            gp_score = np.sum(gp_radii)
            if _is_valid(gp_pos, gp_radii) and gp_score > best_score + 1e-12:
                best_score = gp_score
                best_pos = gp_pos.copy()
                best_radii = gp_radii.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    return np.column_stack([best_pos, best_radii])
>>>>>>> REPLACE
