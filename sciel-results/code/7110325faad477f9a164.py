    best_gap = -1.0
    best_r = 0.15
    
    radii = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]
    
    for r in radii:
        cell = make_cell_4quad(r)
        gap = compute_gap_surrogate(cell)
        if gap > best_gap:
            best_gap = gap
            best_r = r
            
    # Refine
    if time.time() < deadline:
        radii_fine = np.linspace(best_r - 0.02, best_r + 0.02, 11)
        for r in radii_fine:
            if r < 0.05 or r > 0.3: continue
            cell = make_cell_4quad(r)
            gap = compute_gap_surrogate(cell)
            if gap > best_gap:
                best_gap = gap
                best_r = r
                
    return make_cell_4quad(best_r)
