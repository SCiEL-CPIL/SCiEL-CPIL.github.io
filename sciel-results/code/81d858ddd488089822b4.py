<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with FCC lattice
    centres = _fcc_lattice_init()
    
    # Ensure valid
    if centres.shape != (N, 3):
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Phase 1: Coarse optimization with soft-min gradient
    step_size = 0.03
    beta = 30.0
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline - 70.0:
        new_centres, r = _generalized_gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            step_size *= 0.9995
            beta *= 1.003
        else:
            centres = new_centres
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-5:
            break
    
    # Phase 2: Fine optimization with sharper soft-min
    step_size = 0.005
    beta = 100.0
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 40.0:
        new_centres, r = _generalized_gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            step_size *= 0.999
            beta *= 1.005
        else:
            centres = new_centres
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-6:
            break
    
    # Phase 3: Multi-start refinement to escape local optima
    best_centres, best_radius = _multi_start_refine(best_centres, best_radius, deadline, n_starts=5)
    
    # Phase 4: Active-set polish for high precision
    if time.time() < deadline - 5.0:
        polished = _active_set_polish(best_centres, max_iter=200, tol=1e-15)
        polished_r = _compute_radius(polished)
        if polished_r > best_radius:
            best_centres = polished
            best_radius = polished_r
    
    # Final validation: ensure all points are strictly inside the cube
    best_centres = np.clip(best_centres, -0.4999999, 0.4999999)
    
    # Final radius check
    final_radius = _compute_radius(best_centres)
    
    return best_centres
# EVOLVE-BLOCK-END
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with FCC lattice
    centres = _fcc_lattice_init()
    
    # Ensure valid
    if centres.shape != (N, 3):
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Phase 1: Coarse optimization with soft-min gradient
    step_size = 0.03
    beta = 30.0
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 60.0:
        new_centres, r = _generalized_gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            step_size *= 0.999
            beta *= 1.005
        else:
            centres = new_centres
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-5:
            break
    
    # Phase 2: Fine optimization with sharper soft-min
    step_size = 0.005
    beta = 100.0
    no_improve_count = 0
    max_no_improve = 150
    
    while time.time() < deadline - 35.0:
        new_centres, r = _generalized_gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            step_size *= 0.998
            beta *= 1.008
        else:
            centres = new_centres
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-6:
            break
    
    # Phase 3: Multi-start refinement with constraint-targeted push-apart
    best_centres, best_radius = _multi_start_refine(best_centres, best_radius, deadline, n_starts=5)
    
    # Phase 4: Active-set polish for high precision
    if time.time() < deadline - 5.0:
        polished = _active_set_polish(best_centres, max_iter=300, tol=1e-15)
        polished_r = _compute_radius(polished)
        if polished_r > best_radius:
            best_centres = polished
            best_radius = polished_r
    
    # Final validation: ensure all points are strictly inside the cube
    best_centres = np.clip(best_centres, -0.4999999, 0.4999999)
    
    # Final radius check
    final_radius = _compute_radius(best_centres)
    
    return best_centres
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

