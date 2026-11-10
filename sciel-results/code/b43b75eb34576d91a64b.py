<<<<<<< SEARCH
    # Try different initializations
    inits = [
        _grid_init(),
        _hex_init(),
    ]
    
    # Add random perturbations to grid
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = _grid_init() + rng.uniform(-0.02, 0.02, size=(N, 2))
        # Clip to valid range
        perturbed = np.clip(perturbed, -SQUARE_HALF + 0.001, SQUARE_HALF - 0.001)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Run local search
        centers = _local_search(init, beta=0.02, max_iter=150)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Try to improve with smaller beta (sharper)
    if best_centers is not None and time.time() < deadline - 5:
        for beta in [0.01, 0.005]:
            centers = _local_search(best_centers, beta=beta, max_iter=100)
            radius = _compute_radius(centers)
            if radius > best_radius:
                best_radius = radius
                best_centers = centers.copy()
    
    # Final refinement with very small beta
    if best_centers is not None and time.time() < deadline - 2:
        centers = _local_search(best_centers, beta=0.002, max_iter=50)
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to grid
        best_centers = _grid_init()
    
    return best_centers
=======
    # Try different initializations
    inits = [
        _grid_init(),
        _hex_init(),
    ]
    
    # Add 2 random perturbations to grid
    rng = np.random.default_rng(42)
    for _ in range(2):
        perturbed = _grid_init() + rng.uniform(-0.02, 0.02, size=(N, 2))
        # Clip to valid range
        perturbed = np.clip(perturbed, -SQUARE_HALF + 0.001, SQUARE_HALF - 0.001)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline - 20:
            break
        
        # Run local search with reduced iterations
        centers = _local_search(init, beta=0.02, max_iter=100)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # One refinement step with smaller beta if time permits
    if best_centers is not None and time.time() < deadline - 10:
        centers = _local_search(best_centers, beta=0.01, max_iter=50)
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to grid
        best_centers = _grid_init()
    
    return best_centers
>>>>>>> REPLACE
