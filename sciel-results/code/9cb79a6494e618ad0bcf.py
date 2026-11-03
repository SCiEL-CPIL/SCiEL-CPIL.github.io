<<<<<<< SEARCH
def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initial configuration: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    
    best_params = np.array(cells[:N])
    best_score = -np.sum(best_params[:, 2])  # We minimize -sum(r), so score is -obj
    
    # Helper to check validity and get score
    def get_score(params):
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Check boundaries
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5):
            return -np.inf
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-5:
                    return -np.inf
        return np.sum(r)

    best_valid_score = get_score(best_params)
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout:
        # Create a perturbed initial guess
        params = best_params.copy()
        # Add small random noise to positions and radii
        noise_scale = 0.05
        params[:, 0] += rng.uniform(-noise_scale, noise_scale, N)
        params[:, 1] += rng.uniform(-noise_scale, noise_scale, N)
        params[:, 2] += rng.uniform(-noise_scale, noise_scale, N)
        
        # Clip to reasonable bounds to help optimizer
        params[:, 0] = np.clip(params[:, 0], 0.01, 0.99)
        params[:, 1] = np.clip(params[:, 1], 0.01, 0.99)
        params[:, 2] = np.clip(params[:, 2], 0.01, 0.4)
        
        # Optimize
        try:
            res = minimize(
                _penalty_and_objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if res.success:
                candidate = res.x.reshape(N, 3)
                # Ensure radii are positive
                candidate[:, 2] = np.maximum(candidate[:, 2], 1e-6)
                
                # Check validity
                score = get_score(candidate)
                if score > best_valid_score:
                    best_valid_score = score
                    best_params = candidate
        except Exception:
            pass
            
        # Check time
        if time.time() - start_time > timeout:
            break
            
    # Final cleanup: ensure strict validity by shrinking slightly if needed
    # The penalty method might leave tiny violations.
    # We can do a final "fix" pass:
    # 1. If any circle is out of bounds, shrink it or move it.
    # 2. If any circles overlap, shrink the smaller one or move them.
    
    # Simple iterative fix for final result
    final_params = best_params.copy()
    for _ in range(10):
        changed = False
        # Fix boundaries
        for i in range(N):
            min_dist = min(final_params[i, 0], 1 - final_params[i, 0], final_params[i, 1], 1 - final_params[i, 1])
            if final_params[i, 2] > min_dist:
                final_params[i, 2] = min_dist
                changed = True
        
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = final_params[i, 0] - final_params[j, 0]
                dy = final_params[i, 1] - final_params[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = final_params[i, 2] + final_params[j, 2]
                if d < min_dist:
                    # Shrink the smaller radius to resolve overlap
                    # Or scale both? Shrink smaller is safer for sum if one is much larger?
                    # Actually, to maximize sum, we should probably shrink both proportionally or just the smaller one.
                    # Let's shrink the smaller one to fit.
                    if final_params[i, 2] <= final_params[j, 2]:
                        # Reduce r_i so that r_i + r_j = d
                        new_r_i = d - final_params[j, 2]
                        if new_r_i > 0:
                            final_params[i, 2] = new_r_i
                        else:
                            # If d is too small, we must shrink both or move them.
                            # Moving is complex here. Let's just shrink both to half the distance.
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
                    else:
                        new_r_j = d - final_params[i, 2]
                        if new_r_j > 0:
                            final_params[j, 2] = new_r_j
                        else:
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
        if not changed:
            break
            
    return final_params
=======
def construct():
    start_time = time.time()
    
    # Multiple initialization strategies
    rng = np.random.default_rng(42)
    
    # Start with grid initialization
    best_params = _grid_init()
    best_params = _optimize_radii(best_params)
    best_score = np.sum(best_params[:, 2])
    
    # Try random initializations and local search
    while time.time() - start_time < DEADLINE:
        # Generate a random initial configuration
        params = _random_init(rng)
        
        # Optimize radii for this position configuration
        params = _optimize_radii(params)
        
        # Run best-fit local search
        params = _best_fit_local_search(params, max_iter=30)
        
        score = np.sum(params[:, 2])
        if score > best_score + 1e-8:
            best_score = score
            best_params = params.copy()
        
        # Also try perturbing the best solution
        perturbed = best_params.copy()
        idx = rng.integers(0, N)
        perturbed[idx, 0] = np.clip(perturbed[idx, 0] + rng.uniform(-0.03, 0.03), 0.01, 0.99)
        perturbed[idx, 1] = np.clip(perturbed[idx, 1] + rng.uniform(-0.03, 0.03), 0.01, 0.99)
        perturbed = _optimize_radii(perturbed)
        perturbed = _best_fit_local_search(perturbed, max_iter=20)
        
        pert_score = np.sum(perturbed[:, 2])
        if pert_score > best_score + 1e-8:
            best_score = pert_score
            best_params = perturbed.copy()
    
    # Final pass: optimize radii one more time
    best_params = _optimize_radii(best_params)
    
    # Ensure validity with a small safety margin
    for i in range(N):
        min_dist = min(best_params[i, 0], 1 - best_params[i, 0], 
                      best_params[i, 1], 1 - best_params[i, 1])
        best_params[i, 2] = min(best_params[i, 2], min_dist * 0.999)
    
    return best_params
>>>>>>> REPLACE
