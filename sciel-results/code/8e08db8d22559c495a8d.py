<<<<<<< SEARCH
    # Phase 1: Optimize shell radii
    def radial_energy(radii):
        r_in, r_out = radii
        if r_in < 0.5 or r_out < 1.0 or r_out < r_in + 0.3:
            return 1e10
        pos = _build_mackay_with_edge_fractions(r_in, r_out, np.full(30, 0.5))
        if _min_distance(pos) < 0.5:
            return 1e10
        pos_r, e_r = _relax_lbfgs(pos, maxiter=100, gtol=1e-8, ftol=1e-10)
        if _min_distance(pos_r) < 0.5:
            return 1e10
        return e_r
    
    r_init = np.array([1.0, 2.0])
    try:
        result = minimize(radial_energy, r_init, method='Nelder-Mead',
                         options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6,
                                  'adaptive': True})
        if result.fun < radial_energy(r_init):
            r_in_opt, r_out_opt = result.x
        else:
            r_in_opt, r_out_opt = r_init
    except Exception:
        r_in_opt, r_out_opt = r_init
=======
    # Phase 1: Optimize shell radii
    def radial_energy(radii):
        r_in, r_out = radii
        if r_in < 0.4 or r_out < 0.8 or r_out < r_in + 0.2:
            return 1e10 + (max(0, 0.4 - r_in) + max(0, 0.8 - r_out) + max(0, r_in + 0.2 - r_out)) ** 2 * 1e4
        pos = _build_mackay_with_edge_fractions(r_in, r_out, np.full(30, 0.5))
        d_min = _min_distance(pos)
        if d_min < 0.5:
            return 1e10 + (0.5 - d_min) ** 2 * 1e6
        pos_r, e_r = _relax_lbfgs(pos, maxiter=80, gtol=1e-7, ftol=1e-9)
        d_min_r = _min_distance(pos_r)
        if d_min_r < 0.5:
            return 1e10 + (0.5 - d_min_r) ** 2 * 1e6
        return e_r
    
    r_init = np.array([1.0, 2.0])
    e_init = radial_energy(r_init)
    r_in_opt, r_out_opt = r_init
    best_r_e = e_init
    
    try:
        result = minimize(radial_energy, r_init, method='Nelder-Mead',
                         options={'maxiter': 25, 'xatol': 1e-4, 'fatol': 1e-6,
                                  'adaptive': True})
        if result.fun < best_r_e:
            r_in_opt, r_out_opt = result.x
            best_r_e = result.fun
    except Exception:
        pass
    
    # Try a few manual perturbations of radii
    for dr_in, dr_out in [(0.05, 0.05), (-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05), (0.1, 0.1), (-0.1, -0.1)]:
        r_try = np.array([r_in_opt + dr_in, r_out_opt + dr_out])
        e_try = radial_energy(r_try)
        if e_try < best_r_e:
            r_in_opt, r_out_opt = r_try
            best_r_e = e_try
>>>>>>> REPLACE

