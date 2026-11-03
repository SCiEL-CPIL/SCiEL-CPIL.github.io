def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Geodesic Dome (Subdivided Icosahedron)
    try:
        geo_pts = _subdivided_icosahedron(N)
        inits.append(geo_pts)
    except Exception:
        pass
        
    # 2. Fibonacci Sphere
    inits.append(_fibonacci_sphere(N))
    
    # 3. MD Initialized Points (Multiple seeds)
    for seed in [42, 123, 999]:
        try:
            np.random.seed(seed)
            md_pts = _md_init(N, steps=200, dt=0.005, temp=0.5)
            inits.append(md_pts)
        except Exception:
            pass
            
    # Optimization Loop using Riemannian Adam
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 40:  # Leave more time for basin hopping
            break
            
        try:
            # Run Riemannian Adam
            opt_pts = _riemannian_adam(init_pts, learning_rate=0.005, max_iter=500)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
            
    # If we have time, do Basin Hopping
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            remaining_time = DEADLINE - 5 - (time.time() - start_time)
            bh_pts = _basin_hopping(best_pts, max_time=remaining_time, initial_perturb=0.1)
            
            norms = np.linalg.norm(bh_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            bh_pts = bh_pts / norms
            
            e = _log_energy(bh_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = bh_pts.copy()
                
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
