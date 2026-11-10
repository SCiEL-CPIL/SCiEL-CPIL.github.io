<<<<<<< SEARCH
    # Enforce sum constraints and bounds
    best_a = np.clip(best_a, -50.0, 50.0)
    best_b = np.clip(best_b, -50.0, 50.0)
    best_a = best_a + (1.0 - np.sum(best_a)) / 12
    best_b = best_b + (1.0 - np.sum(best_b)) / 12
    
    # Final check: ensure finite values
    if not (np.all(np.isfinite(best_a)) and np.all(np.isfinite(best_b))):
        best_a = np.ones(12) / 12
        best_b = np.ones(12) / 12
    
    return list(best_a), list(best_b)
=======
    # Extract final best coefficients from best_x
    best_a, best_b = normalize_coeffs(best_x)
    
    # Enforce bounds
    best_a = np.clip(best_a, -50.0, 50.0)
    best_b = np.clip(best_b, -50.0, 50.0)
    # Re-normalize after clipping
    best_a, best_b = normalize_coeffs(np.concatenate([best_a, best_b]))
    
    # Final check: ensure finite values
    if not (np.all(np.isfinite(best_a)) and np.all(np.isfinite(best_b))):
        best_a = np.ones(12) / 12
        best_b = np.ones(12) / 12
    
    return list(best_a), list(best_b)
>>>>>>> REPLACE
