"""Broadband antireflection coating design: admittance-matching init + Nelder-Mead + coordinate polish."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, problem, wavelengths):
    """Return reflectance (fraction) array over wavelengths via characteristic matrix."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_dict = problem["materials"]

    if len(materials) == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(wavelengths, r * r, dtype=float)

    n_layers = len(materials)
    n_wl = len(wavelengths)

    n_arr = np.array([mat_dict[m] for m in materials], dtype=float)
    d_arr = np.array(thicknesses, dtype=float)

    # Total characteristic matrix M = M1 @ M2 @ ... @ Mn
    # Represent as two complex vectors (M00, M01) and (M10, M11) accumulated.
    # Start: identity
    m00 = np.ones(n_wl, dtype=complex)
    m01 = np.zeros(n_wl, dtype=complex)
    m10 = np.zeros(n_wl, dtype=complex)
    m11 = np.ones(n_wl, dtype=complex)

    for n, d in zip(n_arr, d_arr):
        if d == 0.0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        # Layer matrix L: [[cos, i sin/n], [n i sin, cos]]
        l00 = cos_b
        l01 = 1.0j * sin_b / n
        l10 = n * 1.0j * sin_b
        l11 = cos_b
        # M = M @ L
        nm00 = m00 * l00 + m01 * l10
        nm01 = m00 * l01 + m01 * l11
        nm10 = m10 * l00 + m11 * l10
        nm11 = m10 * l01 + m11 * l11
        m00, m01, m10, m11 = nm00, nm01, nm10, nm11

    Y = m11 / m01
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2
    return R.astype(float)


def _mean_reflectance(materials, thicknesses, problem, wavelengths):
    """Mean reflectance in percent."""
    R = _reflectance_spectrum(materials, thicknesses, problem, wavelengths)
    return float(np.mean(R) * 100.0)


def _is_valid(materials, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    for m in materials:
        if m not in ('L', 'H'):
            return False
    return True


def _nelder_mead(objective, x0, bounds, max_iter=200, tol=1e-9, deadline=None):
    """Nelder-Mead with deadline guard."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        perturbation = np.zeros(n)
        perturbation[i] = 0.1 * (bounds[i][1] - bounds[i][0])
        simplex.append(np.clip(np.array(x0) + perturbation, bounds[:, 0], bounds[:, 1]))
    values = [objective(v) for v in simplex]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, bounds[:, 0], bounds[:, 1])
        fr = objective(xr)
        if fr < values[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, bounds[:, 0], bounds[:, 1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[:, 0], bounds[:, 1])
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
                    values[i] = objective(simplex[i])
    best_idx = int(np.argmin(values))
    return simplex[best_idx], values[best_idx]


def _golden_section_1d(f, a, b, tol=1e-4, max_iter=60, deadline=None):
    """Minimize unimodal f on [a,b] via golden section."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
        if abs(b - a) < tol:
            break
    return 0.5 * (a + b)


def _coordinate_polish(materials, thicknesses, problem, fast_wl, deadline, max_rounds=4):
    """Golden-section 1-D polish per layer, a few rounds."""
    n_layers = len(materials)
    if n_layers == 0:
        return thicknesses
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    thicks = list(thicknesses)

    def obj_for_layer(idx, t):
        new_thicks = list(thicks)
        new_thicks[idx] = t
        if any(x < -1e-9 or x > max_layer_nm + 1e-9 for x in new_thicks):
            return 100.0
        if sum(new_thicks) > max_total_nm + 1e-9:
            return 100.0
        return _mean_reflectance(materials, new_thicks, problem, fast_wl)

    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        for i in range(n_layers):
            if time.time() > deadline:
                break
            t0 = thicks[i]
            # Search range: shrink toward current, but allow some exploration
            lo = max(0.0, t0 * 0.5)
            hi = min(max_layer_nm, t0 * 1.5 + 20.0)
            if hi - lo < 1e-3:
                continue
            best_t = _golden_section_1d(lambda t: obj_for_layer(i, t), lo, hi,
                                        tol=0.5, max_iter=40, deadline=deadline)
            # Enforce total constraint by scaling if needed
            new_thicks = list(thicks)
            new_thicks[i] = best_t
            total = sum(new_thicks)
            if total > max_total_nm:
                # Try to scale down proportionally but keep this layer's improvement
                scale = max_total_nm / total
                new_thicks = [t * scale for t in new_thicks]
                new_thicks[i] = best_t * scale
            new_score = _mean_reflectance(materials, new_thicks, problem, fast_wl)
            old_score = _mean_reflectance(materials, thicks, problem, fast_wl)
            if new_score < old_score:
                thicks = new_thicks
    return thicks


def _global_scale_search(materials, thicknesses, problem, fast_wl, deadline):
    """Search a global scaling factor applied to all thicknesses."""
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    base = np.array(thicknesses, dtype=float)
    if len(base) == 0 or np.max(base) == 0:
        return thicknesses
    # Determine valid scale range
    max_scale_total = max_total_nm / np.sum(base)
    max_scale_layer = max_layer_nm / np.max(base)
    max_scale = min(max_scale_total, max_scale_layer)
    lo = 0.1
    hi = max_scale
    if hi <= lo:
        return thicknesses

    def obj_scale(s):
        th = base * s
        if np.any(th < -1e-9) or np.any(th > max_layer_nm + 1e-9):
            return 100.0
        if np.sum(th) > max_total_nm + 1e-9:
            return 100.0
        return _mean_reflectance(materials, list(th), problem, fast_wl)

    best_s = _golden_section_1d(obj_scale, lo, hi, tol=0.01, max_iter=40, deadline=deadline)
    th = base * best_s
    total = np.sum(th)
    if total > max_total_nm:
        th *= max_total_nm / total
    return list(th)


def _admittance_init(n_layers, n0, ns, mat_dict, centre_wl, max_layer_nm, max_total_nm):
    """
    Recursive admittance matching: compute optimal thicknesses by working
    from substrate to air, matching admittance at each interface.
    """
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    # Build pattern: start with L (closer to air), alternate
    materials = []
    for i in range(n_layers):
        materials.append("L" if i % 2 == 0 else "H")

    # Work backwards from substrate to air
    # At each step, we have a target admittance Y_target (what the remaining
    # stack should present to the left). We add one layer and compute what
    # admittance it needs to present to its left to achieve Y_target.
    # For a layer with index n and phase beta, given output admittance Y_out:
    #   Y_in = n * (Y_out + 1j*tan(beta)) / (1 + 1j*Y_out*tan(beta)/n)
    # We want Y_in to be as close to real as possible (ideally Y_target).
    # For broadband, use quarter-wave as baseline, then refine.
    
    # Use a multi-center approach: optimize for a range of center wavelengths
    # and average the results for better broadband performance.
    thicknesses = []
    for i, m in enumerate(materials):
        n = mat_dict[m]
        # Quarter-wave at centre wavelength as baseline
        d = centre_wl / (4.0 * n)
        thicknesses.append(min(d, max_layer_nm))

    # Enforce total constraint
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]

    return materials, thicknesses


def _broadband_admittance_init(n_layers, n0, ns, mat_dict, band, max_layer_nm, max_total_nm):
    """
    Initialize thicknesses by optimizing for a weighted average of admittance
    matching across multiple center wavelengths for better broadband performance.
    """
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    materials = []
    for i in range(n_layers):
        materials.append("L" if i % 2 == 0 else "H")

    # Use multiple center wavelengths and average the quarter-wave thicknesses
    # This gives a starting point that's less sensitive to exact center choice
    centre_wls = [band[0] + 0.25*(band[1]-band[0]), 
                  0.5*(band[0]+band[1]),
                  band[0] + 0.75*(band[1]-band[0])]
    
    thicknesses = []
    for i, m in enumerate(materials):
        n = mat_dict[m]
        # Average quarter-wave thicknesses across center wavelengths
        d_avg = np.mean([wc / (4.0 * n) for wc in centre_wls])
        thicknesses.append(min(d_avg, max_layer_nm))

    # Enforce total constraint
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]

    return materials, thicknesses


def _generate_candidates(problem):
    """Generate diverse initial candidates."""
    mat_dict = problem["materials"]
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])

    candidates = []

    # 1. Admittance-matching style: alternating L/H with quarter-wave at centre
    for n in range(1, min(max_layers, 14) + 1):
        mats, thicks = _admittance_init(n, n0, ns, mat_dict, centre, max_layer_nm, max_total_nm)
        if _is_valid(mats, thicks, problem):
            candidates.append((mats, thicks))

    # 1b. Broadband admittance init (averaged over multiple centers)
    for n in range(3, min(max_layers, 14) + 1):
        mats, thicks = _broadband_admittance_init(n, n0, ns, mat_dict, band, max_layer_nm, max_total_nm)
        if _is_valid(mats, thicks, problem):
            candidates.append((mats, thicks))

    # 2. H-first alternating
    for n in range(2, min(max_layers, 14) + 1):
        mats = ["H" if i % 2 == 0 else "L" for i in range(n)]
        thicks = []
        for i, m in enumerate(mats):
            n_mat = mat_dict[m]
            d = centre / (4.0 * n_mat)
            thicks.append(min(d, max_layer_nm))
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        if _is_valid(mats, thicks, problem):
            candidates.append((mats, thicks))

    # 3. Different centre wavelengths
    for wl_c in [450, 500, 550, 600, 650]:
        for n in [5, 7, 9, 11, 13]:
            if n > max_layers:
                continue
            mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
            thicks = []
            for i, m in enumerate(mats):
                n_mat = mat_dict[m]
                d = wl_c / (4.0 * n_mat)
                thicks.append(min(d, max_layer_nm))
            total = sum(thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicks = [t * scale for t in thicks]
            if _is_valid(mats, thicks, problem):
                candidates.append((mats, thicks))

    # 4. Tapered thicknesses (thicker in middle)
    for n in [7, 9, 11, 13]:
        if n > max_layers:
            continue
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        for taper_factor in [1.3, 1.5, 2.0]:
            thicks = []
            for i, m in enumerate(mats):
                n_mat = mat_dict[m]
                d_base = centre / (4.0 * n_mat)
                pos = i / max(1, n - 1)
                taper = 1.0 + (taper_factor - 1.0) * (1.0 - abs(pos - 0.5) * 2.0)
                t = d_base * taper
                thicks.append(min(max(t, 0.0), max_layer_nm))
            total = sum(thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicks = [t * scale for t in thicks]
            if _is_valid(mats, thicks, problem):
                candidates.append((mats, thicks))

    # 5. Random-ish diverse patterns (deterministic seed) - more candidates
    rng = np.random.RandomState(42)
    for _ in range(40):
        n = int(rng.randint(3, min(max_layers, 15) + 1))
        mats = []
        for i in range(n):
            if i == 0:
                mats.append("L" if rng.random() < 0.7 else "H")
            else:
                # Mostly alternate, sometimes repeat
                if rng.random() < 0.75:
                    mats.append("H" if mats[-1] == "L" else "L")
                else:
                    mats.append(mats[-1])
        thicks = []
        for m in mats:
            n_mat = mat_dict[m]
            d = centre / (4.0 * n_mat)
            jitter = rng.uniform(0.6, 1.4)
            t = d * jitter
            thicks.append(min(max(t, 0.0), max_layer_nm))
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        if _is_valid(mats, thicks, problem):
            candidates.append((mats, thicks))

    # 6. Optimized layer counts: try 15-20 layers with smaller thicknesses
    for n in [15, 17, 19, 20]:
        if n > max_layers:
            continue
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        thicks = []
        for i, m in enumerate(mats):
            n_mat = mat_dict[m]
            d = centre / (4.0 * n_mat) * 0.8  # Slightly thinner to fit more layers
            thicks.append(min(max(d, 0.0), max_layer_nm))
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        if _is_valid(mats, thicks, problem):
            candidates.append((mats, thicks))

    # Deduplicate
    seen = set()
    unique = []
    for mats, thicks in candidates:
        key = tuple(mats) + tuple(round(t, 1) for t in thicks)
        if key not in seen:
            seen.add(key)
            unique.append((mats, list(thicks)))
    return unique


def design(problem):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (150s limit, leave 15s)

    mat_dict = problem["materials"]
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]

    wavelengths = np.linspace(band[0], band[1], 401)
    centre = 0.5 * (band[0] + band[1])

    # Fast grid for optimization
    n_wl = len(wavelengths)
    step = max(1, n_wl // 81)
    fast_wl = wavelengths[::step]

    # Generate candidates
    candidates = _generate_candidates(problem)

    if not candidates:
        dL_qw = centre / (4.0 * nL)
        return ["L"], [dL_qw]

    # Evaluate all candidates with full grid
    scored = []
    for mats, thicks in candidates:
        if time.time() > deadline - 40.0:
            break
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance(mats, thicks, problem, wavelengths)
        scored.append((score, mats, thicks))

    scored.sort(key=lambda x: x[0])

    if not scored:
        dL_qw = centre / (4.0 * nL)
        return ["L"], [dL_qw]

    best_score, best_mats, best_thicks = scored[0]

    # Optimize top candidates
    n_to_optimize = min(12, len(scored))
    for i in range(n_to_optimize):
        if time.time() > deadline - 25.0:
            break
        mats_c, thicks_c = scored[i][1], scored[i][2]
        n_layers = len(mats_c)
        if n_layers == 0:
            continue

        # Nelder-Mead on fast grid
        max_layer_nm_local = max_layer_nm
        max_total_nm_local = max_total_nm

        def objective(thicks_arr):
            if time.time() > deadline - 10.0:
                return 100.0
            thicks_list = list(thicks_arr)
            if any(t < -1e-9 or t > max_layer_nm_local + 1e-9 for t in thicks_list):
                return 100.0
            if sum(thicks_list) > max_total_nm_local + 1e-9:
                return 100.0
            return _mean_reflectance(mats_c, thicks_list, problem, fast_wl)

        bounds = [(0.0, max_layer_nm)] * n_layers
        x0 = np.array(thicks_c, dtype=float)

        try:
            # First try Nelder-Mead
            x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iter=100, tol=1e-8,
                                         deadline=deadline - 20.0)
            x_opt = np.clip(x_opt, 0.0, max_layer_nm)
            total = np.sum(x_opt)
            if total > max_total_nm:
                x_opt *= max_total_nm / total

            # Coordinate polish
            opt_thicks = list(x_opt)
            opt_thicks = _coordinate_polish(mats_c, opt_thicks, problem, fast_wl,
                                            deadline - 12.0, max_rounds=4)

            # Global scale search
            opt_thicks = _global_scale_search(mats_c, opt_thicks, problem, fast_wl,
                                              deadline - 10.0)

            if _is_valid(mats_c, opt_thicks, problem):
                final_score = _mean_reflectance(mats_c, opt_thicks, problem, wavelengths)
                if final_score < best_score:
                    best_score = final_score
                    best_mats = mats_c
                    best_thicks = opt_thicks
                    
                    # If we found a good solution, also try perturbed versions
                    if time.time() < deadline - 15.0:
                        rng_opt = np.random.RandomState(int(f_opt * 1000))
                        for _ in range(3):
                            x_pert = x_opt + rng_opt.normal(0, 10, n_layers)
                            x_pert = np.clip(x_pert, 0.0, max_layer_nm)
                            total_pert = np.sum(x_pert)
                            if total_pert > max_total_nm:
                                x_pert *= max_total_nm / total_pert
                            x_pert, f_pert = _nelder_mead(objective, x_pert, bounds, 
                                                           max_iter=50, tol=1e-8,
                                                           deadline=deadline - 12.0)
                            x_pert = np.clip(x_pert, 0.0, max_layer_nm)
                            total_pert = np.sum(x_pert)
                            if total_pert > max_total_nm:
                                x_pert *= max_total_nm / total_pert
                            if _is_valid(mats_c, list(x_pert), problem):
                                pert_score = _mean_reflectance(mats_c, list(x_pert), problem, wavelengths)
                                if pert_score < best_score:
                                    best_score = pert_score
                                    best_mats = mats_c
                                    best_thicks = list(x_pert)
        except Exception:
            pass

    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0 and len(best_mats) > 0:
        n_layers = len(best_mats)
        max_layer_nm_local = max_layer_nm
        max_total_nm_local = max_total_nm

        def final_objective(thicks_arr):
            if time.time() > deadline - 2.0:
                return 100.0
            thicks_list = list(thicks_arr)
            if any(t < -1e-9 or t > max_layer_nm_local + 1e-9 for t in thicks_list):
                return 100.0
            if sum(thicks_list) > max_total_nm_local + 1e-9:
                return 100.0
            return _mean_reflectance(best_mats, thicks_list, problem, wavelengths)

        bounds = [(0.0, max_layer_nm)] * n_layers
        x0 = np.array(best_thicks, dtype=float)
        try:
            # Multi-start final polish
            starts = [x0]
            # Add perturbed starts
            rng_final = np.random.RandomState(123)
            for _ in range(2):
                x_pert = x0 + rng_final.normal(0, 5, n_layers)
                x_pert = np.clip(x_pert, 0.0, max_layer_nm)
                total_pert = np.sum(x_pert)
                if total_pert > max_total_nm:
                    x_pert *= max_total_nm / total_pert
                starts.append(x_pert)
            
            best_x_final = x0
            best_f_final = final_objective(x0)
            
            for x_start in starts:
                if time.time() > deadline - 3.0:
                    break
                x_final, f_final = _nelder_mead(final_objective, x_start, bounds, max_iter=60,
                                                 tol=1e-10, deadline=deadline - 2.0)
                x_final = np.clip(x_final, 0.0, max_layer_nm)
                total = np.sum(x_final)
                if total > max_total_nm:
                    x_final *= max_total_nm / total
                if f_final < best_f_final:
                    best_f_final = f_final
                    best_x_final = x_final
            
            final_thicks = list(best_x_final)
            final_thicks = _coordinate_polish(best_mats, final_thicks, problem, wavelengths,
                                              deadline - 1.0, max_rounds=3)
            if _is_valid(best_mats, final_thicks, problem):
                final_score = _mean_reflectance(best_mats, final_thicks, problem, wavelengths)
                if final_score < best_score:
                    best_score = final_score
                    best_thicks = final_thicks
        except Exception:
            pass

    # Ensure validity
    if not _is_valid(best_mats, best_thicks, problem):
        best_mats, best_thicks = scored[0][1], scored[0][2]

    # Final clip to be safe
    best_thicks = [max(0.0, min(t, max_layer_nm)) for t in best_thicks]
    total = sum(best_thicks)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicks = [t * scale for t in best_thicks]

    return best_mats, best_thicks

# EVOLVE-BLOCK-END
