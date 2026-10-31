<<<<<<< SEARCH
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    
    # 1. Deterministic "38 closest" (first 19 outer atoms)
    # This is equivalent to the previous program's approach
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    # Scale
    diff = det_start[:, np.newaxis, :] - det_start[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        det_start *= scale
    starts.append(det_start)
    
    # 2. Random subsets of outer atoms
    num_random_starts = 40
    for i in range(num_random_starts):
        if time.time() > deadline - 60.0:
            break
        # Pick 19 random outer atoms
        idx = rng.choice(len(outer), size=19, replace=False)
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        
        # Scale
        diff = start_pos[:, np.newaxis, :] - start_pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        nn = np.sqrt(np.min(r2, axis=1))
        avg_nn = np.mean(nn)
        if avg_nn > 1e-12:
            scale = (2.0 ** (1.0 / 6.0)) / avg_nn
            start_pos *= scale
            
        starts.append(start_pos)
        
    # 3. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf

    for start in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            p, e = _relax(start, max_iter=300)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Final polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-12)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
=======
def _scale_to_lj(pos):
    """Scale positions so nearest-neighbor distance is 2^(1/6)."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    min_nn = np.min(nn)
    if min_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / min_nn
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin (5s buffer before 140s kill)

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    
    # 1. Deterministic: all 19 core + 19 outer (first 19 by index)
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    det_start = _scale_to_lj(det_start)
    starts.append(det_start)
    
    # 2. Another deterministic: all 19 core + last 19 outer
    det_start2 = np.vstack([core, outer[5:]])  # different 19 outer atoms
    det_start2 -= np.mean(det_start2, axis=0)
    det_start2 = _scale_to_lj(det_start2)
    starts.append(det_start2)
    
    # 3. Random subsets of outer atoms with optional perturbation
    num_random_starts = 50
    for i in range(num_random_starts):
        if time.time() > deadline - 80.0:
            break
        # Pick 19 random outer atoms
        idx = rng.choice(len(outer), size=19, replace=False)
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        start_pos = _scale_to_lj(start_pos)
        
        # Add small random perturbation for diversity
        if i % 3 == 0:
            start_pos += rng.normal(0, 0.05, size=start_pos.shape)
        
        starts.append(start_pos)
        
    # 4. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass
    
    # 5. Pure random starts
    for i in range(5):
        if time.time() > deadline - 80.0:
            break
        rand_pos = rng.normal(0, 1.5, size=(N, 3))
        rand_pos -= np.mean(rand_pos, axis=0)
        rand_pos = _scale_to_lj(rand_pos)
        starts.append(rand_pos)

    best_pos = None
    best_energy = np.inf

    for i, start in enumerate(starts):
        if time.time() > deadline - 40.0:
            break
        try:
            p, e = _relax(start, max_iter=300)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Final high-precision polish
    if time.time() < deadline - 15.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # Second polish pass with even tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
>>>>>>> REPLACE

