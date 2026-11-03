<<<<<<< SEARCH
    # Sort initializations by quick energy and take top ones
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    top_candidates = scored_inits[:15]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 10.0:
            break
        
        # Phase 1: Coarse optimization
        optimized = _optimize(init_pts, deadline, max_iter=500, ftol=1e-10, gtol=1e-6, maxls=30)
        
        # Phase 2: Fine optimization
        if time.time() < deadline - 5.0:
            optimized = _optimize(optimized, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 2.0:
        refined = _optimize(best_pts, deadline, max_iter=2000, ftol=1e-16, gtol=1e-13, maxls=50)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
=======
    # Sort initializations by quick energy and take top ones
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 40.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    top_candidates = scored_inits[:18]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 15.0:
            break
        
        # Phase 1: Coarse optimization
        optimized = _optimize(init_pts, deadline, max_iter=400, ftol=1e-10, gtol=1e-6, maxls=25)
        
        # Phase 2: Fine optimization
        if time.time() < deadline - 10.0:
            optimized = _optimize(optimized, deadline, max_iter=1500, ftol=1e-14, gtol=1e-11, maxls=40)
        
        # Phase 3: High-precision gradient polish
        if time.time() < deadline - 5.0:
            optimized = _gradient_polish(optimized, deadline, max_iter=3000, step_scale=1e-5)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Final high-precision polish on the global best
    if best_pts is not None and time.time() < deadline - 3.0:
        # One more L-BFGS-B pass with very tight tolerance
        refined = _optimize(best_pts, deadline, max_iter=1000, ftol=1e-16, gtol=1e-14, maxls=50)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
        
        # Final gradient polish
        if time.time() < deadline - 1.0:
            polished = _gradient_polish(best_pts, deadline, max_iter=5000, step_scale=1e-6)
            e_polished = _precise_energy(polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pts = polished
>>>>>>> REPLACE
