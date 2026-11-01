<<<<<<< SEARCH
def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    rng = np.random.default_rng(42)
    
    # Step 1: Optimize the shell radii
    initial_radii = np.array([1.135, 1.162, 1.283])
    optimal_radii = _optimize_radii(initial_radii)
    
    # Construct base geometry with optimized radii
    base = _mackay_icosahedron_55(*optimal_radii)
    
    # Multi-start stochastic search
    best_energy = float('inf')
    best_pos = base.copy()
    
    # First, try the exact optimized geometry
    relaxed = _relax(base)
    if _is_valid(relaxed):
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Try diverse perturbation strategies
    # Strategy 1: Gaussian noise at various scales
    noise_scales = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
    for scale in noise_scales:
        noise = rng.normal(0, scale, size=base.shape)
        pos = base + noise
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Strategy 2: Radial perturbations (scale each shell differently)
    for trial in range(20):
        r1_pert = optimal_radii[0] * (1 + rng.normal(0, 0.02))
        r2_pert = optimal_radii[1] * (1 + rng.normal(0, 0.02))
        r3_pert = optimal_radii[2] * (1 + rng.normal(0, 0.02))
        pos = _mackay_icosahedron_55(r1_pert, r2_pert, r3_pert)
        
        # Add small Gaussian noise
        noise = rng.normal(0, 0.01, size=pos.shape)
        pos = pos + noise
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Strategy 3: Angular perturbations (rotate individual atoms slightly)
    for trial in range(15):
        pos = base.copy()
        # Apply small random rotations to each atom
        for i in range(1, len(pos)):
            angle = rng.normal(0, 0.01)
            axis = rng.normal(size=3)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            # Rodrigues rotation
            c = np.cos(angle)
            s = np.sin(angle)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
            pos[i] = R @ pos[i]
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Final fine-tuning pass with tighter tolerances
    if best_energy < float('inf'):
        refined = _relax(best_pos, maxiter=30000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    return best_pos
=======
def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    rng = np.random.default_rng(42)
    
    # Step 1: Optimize the shell radii with multi-start
    initial_radii = np.array([1.135, 1.162, 1.283])
    optimal_radii = _optimize_radii(initial_radii)
    
    # Construct base geometry with optimized radii
    base = _mackay_icosahedron_55(*optimal_radii)
    
    # Multi-start stochastic search
    best_energy = float('inf')
    best_pos = base.copy()
    
    # First, try the exact optimized geometry with tight relaxation
    relaxed = _relax(base, maxiter=20000, ftol=1e-13, gtol=1e-11)
    if _is_valid(relaxed):
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Try a few diverse perturbation strategies (limited to save time)
    # Strategy 1: Gaussian noise at small scales
    noise_scales = [0.002, 0.005, 0.01, 0.02]
    for scale in noise_scales:
        noise = rng.normal(0, scale, size=base.shape)
        pos = base + noise
        
        relaxed = _relax(pos, maxiter=15000, ftol=1e-12, gtol=1e-10)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Strategy 2: Radial perturbations
    for trial in range(8):
        r1_pert = optimal_radii[0] * (1 + rng.normal(0, 0.01))
        r2_pert = optimal_radii[1] * (1 + rng.normal(0, 0.01))
        r3_pert = optimal_radii[2] * (1 + rng.normal(0, 0.01))
        pos = _mackay_icosahedron_55(r1_pert, r2_pert, r3_pert)
        
        # Add small Gaussian noise
        noise = rng.normal(0, 0.005, size=pos.shape)
        pos = pos + noise
        
        relaxed = _relax(pos, maxiter=15000, ftol=1e-12, gtol=1e-10)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Final high-precision polish: iterate relaxation until convergence
    if best_energy < float('inf'):
        for _ in range(3):
            refined = _relax(best_pos, maxiter=50000, ftol=1e-15, gtol=1e-13)
            if _is_valid(refined):
                e = _lj_energy(refined)
                if e < best_energy:
                    best_energy = e
                    best_pos = refined.copy()
                else:
                    break
            else:
                break
    
    return best_pos
>>>>>>> REPLACE
