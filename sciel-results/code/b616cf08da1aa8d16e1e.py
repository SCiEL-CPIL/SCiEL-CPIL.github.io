<<<<<<< SEARCH
    # Initialize with the best seed
    for seed in seeds:
        s, outer_params = _fit_outer_hex(seed, np.zeros(12))
        if s < best_score:
            best_score = s
            best_params = np.concatenate([seed.flatten(), outer_params])
=======
    # Initialize with the best seed
    for seed in seeds:
        # Ensure seed is exactly 12x2
        if seed.shape[0] > 12:
            seed = seed[:12]
        elif seed.shape[0] < 12:
            # Pad with center if too small (should not happen with fixed seeds)
            pad = np.zeros((12 - seed.shape[0], 2))
            seed = np.vstack([seed, pad])
        
        s, outer_params = _fit_outer_hex(seed, np.zeros(12))
        if s < best_score:
            best_score = s
            best_params = np.concatenate([seed.flatten(), outer_params])
>>>>>>> REPLACE
