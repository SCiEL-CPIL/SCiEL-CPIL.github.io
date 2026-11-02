<<<<<<< SEARCH
def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # 5s safety margin before 150s kill
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 15.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Simulated annealing to escape local minima
    remaining = _remaining_time()
    if remaining > 12.0:
        sa_budget = min(8.0, remaining * 0.4)
        xyz_sa, score_sa = _sa_search(best_xyz, time_budget=sa_budget, seed=42)
        if _validate(xyz_sa, tol=1e-6):
            score_sa = np.sum(xyz_sa[:, 2])
            if score_sa > best_score:
                best_score = score_sa
                best_xyz = xyz_sa
    
    # Local refinement from the best solution found
    remaining = _remaining_time()
    if remaining > 12.0:
        rng = np.random.default_rng(42)
        # Try multiple perturbation scales
        for scale in [0.005, 0.01, 0.02, 0.03]:
            if remaining < 5.0:
                break
            # Perturb the best solution
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            # Do not perturb radii too much
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()
    
    # Final high-precision polish
    remaining = _remaining_time()
    if remaining > 6.0:
        polish_budget = min(5.0, remaining - 3.0)
        xyz_polish, score_polish = _final_polish(best_xyz, time_budget=polish_budget)
        if _validate(xyz_polish, tol=1e-7):
            score_polish = np.sum(xyz_polish[:, 2])
            if score_polish > best_score:
                best_score = score_polish
                best_xyz = xyz_polish
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz
=======
def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 143.0  # 7s safety margin before 150s kill
    
    best_xyz = _initial_corner_cluster()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_corner_cluster(),
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_corner_weighted(2),
    ]
    
    # Phase A: Optimize from each initial config
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 18.0:
            break
        
        # Use trust-constr for first few, L-BFGS-B for rest
        if inits.index(xyz0) < 3:
            xyz_opt, score = _optimize_trust(xyz0, time_budget=min(14.0, remaining - 4.0))
        else:
            xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt, tol=1e-6):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix, tol=1e-6):
                    break
                xyz_fix[:, 2] *= 0.97
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt, tol=1e-6):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt.copy()
    
    # Phase B: Delaunay contact polish on best solution
    remaining = _remaining_time()
    if remaining > 15.0:
        polish_budget = min(10.0, remaining - 5.0)
        xyz_pol, score_pol = _delaunay_contact_polish(best_xyz, time_budget=polish_budget)
        if _validate(xyz_pol, tol=1e-7):
            score_pol = np.sum(xyz_pol[:, 2])
            if score_pol > best_score:
                best_score = score_pol
                best_xyz = xyz_pol.copy()
    
    # Phase C: Simulated annealing to escape local minima
    remaining = _remaining_time()
    if remaining > 12.0:
        sa_budget = min(8.0, remaining * 0.4)
        xyz_sa, score_sa = _sa_search(best_xyz, time_budget=sa_budget, seed=42)
        if _validate(xyz_sa, tol=1e-6):
            score_sa = np.sum(xyz_sa[:, 2])
            if score_sa > best_score:
                best_score = score_sa
                best_xyz = xyz_sa.copy()
    
    # Phase D: Local refinement with perturbations
    remaining = _remaining_time()
    if remaining > 15.0:
        rng = np.random.default_rng(42)
        for scale in [0.005, 0.01, 0.02]:
            remaining = _remaining_time()
            if remaining < 8.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt, tol=1e-6):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix, tol=1e-6):
                        break
                    xyz_fix[:, 2] *= 0.97
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt, tol=1e-6):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt.copy()
    
    # Phase E: Second Delaunay polish
    remaining = _remaining_time()
    if remaining > 12.0:
        polish_budget = min(8.0, remaining - 4.0)
        xyz_pol2, score_pol2 = _delaunay_contact_polish(best_xyz, time_budget=polish_budget)
        if _validate(xyz_pol2, tol=1e-7):
            score_pol2 = np.sum(xyz_pol2[:, 2])
            if score_pol2 > best_score:
                best_score = score_pol2
                best_xyz = xyz_pol2.copy()
    
    # Phase F: Final high-precision polish
    remaining = _remaining_time()
    if remaining > 8.0:
        polish_budget = min(6.0, remaining - 3.0)
        xyz_polish, score_polish = _final_polish(best_xyz, time_budget=polish_budget)
        if _validate(xyz_polish, tol=1e-7):
            score_polish = np.sum(xyz_polish[:, 2])
            if score_polish > best_score:
                best_score = score_polish
                best_xyz = xyz_polish.copy()
    
    # Final validation with fallback
    if not _validate(best_xyz, tol=1e-7):
        # Try to fix
        for _ in range(100):
            if _validate(best_xyz, tol=1e-7):
                break
            best_xyz[:, 2] *= 0.98
            best_xyz = _fix_boundaries(best_xyz)
        if not _validate(best_xyz, tol=1e-7):
            best_xyz = _initial_grid()
    
    return best_xyz
>>>>>>> REPLACE
