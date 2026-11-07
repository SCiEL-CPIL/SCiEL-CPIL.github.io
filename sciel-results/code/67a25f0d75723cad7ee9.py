<<<<<<< SEARCH
    # Phase 1: Relax each template directly
    for name, template in templates:
        if _time_left() < 10.0:
            break
        try:
            energy, positions = relax(template, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Phase 2: Basin-hopping from each template
    for name, template in templates:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = _basin_hop(
                template, potential, kind, rng,
                max_kicks=10, kick_scale=0.04, time_budget=10.0
            )
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Phase 3: Perturbed templates + relax
    perturbation_scales = [0.03, 0.06, 0.1, 0.15]
    for name, template in templates[:4]:  # Limit to first 4 templates
        if _time_left() < 10.0:
            break
        for scale in perturbation_scales:
            if _time_left() < 5.0:
                break
            pert = _perturb(template, rng, scale)
            if not _is_valid(pert):
                continue
            try:
                energy, positions = relax(pert, potential, kind)
                if np.isfinite(energy) and _is_valid(positions):
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except Exception:
                pass

    # Phase 4: More random starts if time remains
    if _time_left() > 20.0:
        for _ in range(5):
            if _time_left() < 10.0:
                break
            rc = _make_random_compact(n, rng)
            if rc is not None and _is_valid(rc):
                try:
                    energy, positions = _basin_hop(
                        rc, potential, kind, rng,
                        max_kicks=8, kick_scale=0.05, time_budget=8.0
                    )
                    if np.isfinite(energy) and _is_valid(positions):
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
                except Exception:
                    pass
=======
    # Phase 1: Relax each template directly
    for name, template in templates:
        if _time_left() < 8.0:
            break
        try:
            energy, positions = relax(template, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Phase 2: Basin-hopping from best templates (prioritize structured ones)
    # Sort templates by their direct relaxation energy
    template_energies = []
    for name, template in templates:
        try:
            energy, _ = relax(template, potential, kind)
            template_energies.append((energy, name, template))
        except Exception:
            template_energies.append((np.inf, name, template))
    template_energies.sort(key=lambda x: x[0])
    
    for energy, name, template in template_energies:
        if _time_left() < 12.0:
            break
        try:
            bh_energy, bh_positions = _basin_hop(
                template, potential, kind, rng,
                max_kicks=12, kick_scale=0.03, time_budget=8.0
            )
            if np.isfinite(bh_energy) and _is_valid(bh_positions):
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions
        except Exception:
            pass

    # Phase 3: Perturbed best templates + relax
    perturbation_scales = [0.02, 0.04, 0.08, 0.12, 0.2]
    for energy, name, template in template_energies[:3]:  # Top 3 templates
        if _time_left() < 8.0:
            break
        for scale in perturbation_scales:
            if _time_left() < 4.0:
                break
            pert = _perturb(template, rng, scale)
            if not _is_valid(pert):
                continue
            try:
                e, pos = relax(pert, potential, kind)
                if np.isfinite(e) and _is_valid(pos):
                    if e < best_energy:
                        best_energy = e
                        best_positions = pos
            except Exception:
                pass

    # Phase 4: Additional random starts with basin hopping if time remains
    if _time_left() > 15.0:
        for _ in range(4):
            if _time_left() < 8.0:
                break
            rc = _make_random_compact(n, rng)
            if rc is not None and _is_valid(rc):
                try:
                    bh_energy, bh_pos = _basin_hop(
                        rc, potential, kind, rng,
                        max_kicks=10, kick_scale=0.04, time_budget=6.0
                    )
                    if np.isfinite(bh_energy) and _is_valid(bh_pos):
                        if bh_energy < best_energy:
                            best_energy = bh_energy
                            best_positions = bh_pos
                except Exception:
                    pass
>>>>>>> REPLACE

