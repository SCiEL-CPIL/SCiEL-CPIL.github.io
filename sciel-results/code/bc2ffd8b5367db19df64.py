<<<<<<< SEARCH
def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Multi-start local search
    while time.time() - t0 < DEADLINE:
        # Perturb best solution
        jitter = 0.02 * rng.standard_normal((N, 2))
        cand = best + jitter
        
        # Project back into unit circle if needed
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 10.0)
        
        cand, r = _local_search(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
=======
def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Phase 1: Multi-start local search (discrete hill climbing)
    while time.time() - t0 < DEADLINE * 0.6:
        # Perturb best solution
        jitter = 0.02 * rng.standard_normal((N, 2))
        cand = best + jitter
        
        # Project back into unit circle if needed
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE * 0.6 - (time.time() - t0)
        if t_left < 3.0:
            break
        t_alloc = min(t_left - 1.0, 8.0)
        
        cand, r = _local_search(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 2: Smooth soft-min polish with L-BFGS-B
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 10.0:
        t_polish = min(t_left - 5.0, 30.0)
        best, best_r = _smooth_polish(best, t_polish, rng)
        
        # After polish, do a few more local search passes to fine-tune
        while time.time() - t0 < DEADLINE - 5.0:
            jitter = 0.005 * rng.standard_normal((N, 2))
            cand = best + jitter
            rad = np.linalg.norm(cand, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)
            
            t_left = DEADLINE - 5.0 - (time.time() - t0)
            if t_left < 2.0:
                break
            t_alloc = min(t_left - 1.0, 5.0)
            
            cand, r = _local_search(cand, t_alloc, rng)
            if r > best_r:
                best_r = r
                best = cand
            else:
                # Try a small polish again
                if time.time() - t0 < DEADLINE - 8.0:
                    best, best_r = _smooth_polish(best, 5.0, rng)

    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
>>>>>>> REPLACE
