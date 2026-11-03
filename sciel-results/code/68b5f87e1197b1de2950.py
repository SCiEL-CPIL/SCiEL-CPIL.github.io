<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: L-BFGS from multiple initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        x0 = init.flatten()
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50}
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 2: Gradient descent refinement
    if time.time() < deadline - 20.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, lr=0.005, steps=3000, deadline=deadline - 20.0
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()
    
    # Phase 3: Basin hopping for escape from local minima
    if time.time() < deadline - 10.0:
        temperature = 0.01
        for step in range(50):
            if time.time() > deadline - 10.0:
                break
            
            bh_points, bh_energy = _basin_hopping_step(
                best_points, best_energy, temperature, deadline=deadline - 10.0
            )
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_points = bh_points.copy()
                temperature *= 0.95
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 5: Ultra-precise final polish
    if time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_points
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: L-BFGS from multiple initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        p_res, e_check = _lbfgs_minimize(
            init, maxiter=2000, ftol=1e-12, gtol=1e-9, deadline=deadline - 30.0
        )
        if e_check < best_energy:
            best_energy = e_check
            best_points = p_res.copy()
    
    # Phase 2: Basin hopping for escape from local minima
    if time.time() < deadline - 15.0:
        temperature = 0.05
        seed_counter = 0
        for step in range(80):
            if time.time() > deadline - 15.0:
                break
            
            seed_counter += 1
            bh_points, bh_energy = _basin_hopping_step(
                best_points, best_energy, temperature, seed=seed_counter,
                deadline=deadline - 15.0
            )
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_points = bh_points.copy()
                temperature *= 0.92
            else:
                temperature *= 0.97
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        p_res, e_check = _lbfgs_minimize(
            best_points, maxiter=5000, ftol=1e-14, gtol=1e-11, deadline=deadline - 5.0
        )
        if e_check < best_energy:
            best_energy = e_check
            best_points = p_res.copy()
    
    # Phase 4: Ultra-precise final polish
    if time.time() < deadline - 2.0:
        p_res, e_check = _lbfgs_minimize(
            best_points, maxiter=3000, ftol=1e-15, gtol=1e-12, deadline=deadline - 2.0
        )
        if e_check < best_energy:
            best_energy = e_check
            best_points = p_res.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_points
>>>>>>> REPLACE

