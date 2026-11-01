<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Try multiple initializations and pick the best result
    best_x = None
    best_energy = float('inf')
    
    initializations = [
        ("halton", _halton_init(N)),
        ("fibonacci", _fibonacci_init(N)),
    ]
    
    # Add a few random initializations
    np.random.seed(42)
    for _ in range(2):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    for name, x0 in initializations:
        if time.time() > deadline:
            break
        
        x0 = _project_to_sphere(x0)
        
        # Run BFGS optimization
        try:
            result = minimize(
                _energy_and_grad,
                x0.flatten(),
                jac=True,
                method='BFGS',
                options={'maxiter': 2000, 'gtol': 1e-10, 'disp': False}
            )
            
            x_opt = _project_to_sphere(result.x.reshape(N, 3))
            
            # Verify and compute final energy
            energy, _ = _energy_and_grad(x_opt)
            
            # Check validity
            if not np.all(np.isfinite(x_opt)):
                continue
            if np.any(np.linalg.norm(x_opt, axis=1) < 1e-15):
                continue
            
            # Check minimum distance
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_dist_sq = np.min(dist_sq)
            if min_dist_sq < (1e-9) ** 2:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_x = x_opt.copy()
                
        except Exception:
            continue
    
    if best_x is None:
        # Fallback to fibonacci
        best_x = _fibonacci_init(N)
    
    return _project_to_sphere(best_x)
=======
def _polish_l_bfgs_b(x0, max_iter=5000, gtol=1e-12):
    """Polish a configuration using L-BFGS-B with tight tolerances."""
    x0_flat = x0.flatten()
    
    def obj_and_grad(flat_x):
        x = flat_x.reshape(N, 3)
        # Project to sphere for evaluation
        x_norm = _project_to_sphere(x)
        e, g = _energy_and_grad_flat(x_norm.flatten())
        # The gradient needs to be projected tangent to the sphere
        # For a point on the sphere, the constraint is ||x||=1
        # The projected gradient is g - (g.x)*x
        g_proj = g.reshape(N, 3) - np.sum(g.reshape(N, 3) * x_norm, axis=1, keepdims=True) * x_norm
        return e, g_proj.flatten()
    
    result = minimize(
        obj_and_grad,
        x0_flat,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
    )
    
    x_opt = _project_to_sphere(result.x.reshape(N, 3))
    return x_opt


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Try multiple initializations and pick the best result
    best_x = None
    best_energy = float('inf')
    
    initializations = [
        ("halton", _halton_init(N)),
        ("fibonacci", _fibonacci_init(N)),
    ]
    
    # Add random initializations
    np.random.seed(42)
    for _ in range(3):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    # Add farthest point init
    try:
        initializations.append(("fps", _farthest_point_init(N)))
    except Exception:
        pass
    
    for name, x0 in initializations:
        if time.time() > deadline:
            break
        
        x0 = _project_to_sphere(x0)
        
        # Run L-BFGS-B optimization (faster than BFGS for this problem)
        try:
            result = minimize(
                _energy_and_grad_flat,
                x0.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'gtol': 1e-10, 'disp': False}
            )
            
            x_opt = _project_to_sphere(result.x.reshape(N, 3))
            
            # Verify and compute final energy
            energy, _ = _energy_and_grad_flat(x_opt.flatten())
            
            # Check validity
            if not np.all(np.isfinite(x_opt)):
                continue
            if np.any(np.linalg.norm(x_opt, axis=1) < 1e-15):
                continue
            
            # Check minimum distance
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_dist_sq = np.min(dist_sq)
            if min_dist_sq < (1e-9) ** 2:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_x = x_opt.copy()
                
        except Exception:
            continue
    
    # Basin hopping with reheat to escape local minima
    if best_x is not None and time.time() < deadline - 15.0:
        rng = np.random.RandomState(123)
        current_x = best_x.copy()
        current_energy = best_energy
        alpha = 0.1  # perturbation scale
        reheat_every = 10
        
        for iteration in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb
            noise = rng.randn(N, 3) * alpha
            perturbed = current_x + noise
            perturbed = _project_to_sphere(perturbed)
            
            # Optimize the perturbed configuration
            try:
                result = minimize(
                    _energy_and_grad_flat,
                    perturbed.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-10, 'disp': False}
                )
                
                x_new = _project_to_sphere(result.x.reshape(N, 3))
                energy_new, _ = _energy_and_grad_flat(x_new.flatten())
                
                # Check validity
                if not np.all(np.isfinite(x_new)):
                    continue
                if np.any(np.linalg.norm(x_new, axis=1) < 1e-15):
                    continue
                
                diff = x_new[:, np.newaxis, :] - x_new[np.newaxis, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                min_dist_sq = np.min(dist_sq)
                if min_dist_sq < (1e-9) ** 2:
                    continue
                
                # Accept if better or with Metropolis criterion
                if energy_new < current_energy:
                    current_x = x_new
                    current_energy = energy_new
                    alpha *= 0.9  # cool down
                    if energy_new < best_energy:
                        best_energy = energy_new
                        best_x = x_new.copy()
                else:
                    # Metropolis acceptance
                    delta_e = energy_new - current_energy
                    if delta_e < 0.1 * alpha:  # simplified acceptance
                        current_x = x_new
                        current_energy = energy_new
                        alpha *= 0.95
                    else:
                        alpha = max(0.05, alpha * 0.9)
                
                # Reheat every reheat_every iterations
                if iteration % reheat_every == reheat_every - 1:
                    alpha = max(alpha, 0.08)
                    
            except Exception:
                continue
    
    # Final high-precision polish pass
    if best_x is not None and time.time() < deadline - 3.0:
        try:
            x_polished = _polish_l_bfgs_b(best_x, max_iter=10000, gtol=1e-14)
            energy_polished, _ = _energy_and_grad_flat(x_polished.flatten())
            
            if np.all(np.isfinite(x_polished)):
                if np.all(np.linalg.norm(x_polished, axis=1) > 1e-15):
                    diff = x_polished[:, np.newaxis, :] - x_polished[np.newaxis, :, :]
                    dist_sq = np.sum(diff * diff, axis=2)
                    np.fill_diagonal(dist_sq, np.inf)
                    min_dist_sq = np.min(dist_sq)
                    if min_dist_sq >= (1e-9) ** 2:
                        if energy_polished < best_energy:
                            best_energy = energy_polished
                            best_x = x_polished.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback to fibonacci
        best_x = _fibonacci_init(N)
    
    return _project_to_sphere(best_x)
>>>>>>> REPLACE
