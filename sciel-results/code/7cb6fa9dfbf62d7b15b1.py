"""Broadband antireflection coating design: fast coordinate-descent with smart init."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _mean_reflectance_fast(materials, thicknesses, problem, wavelengths):
    """Vectorized mean reflectance (percent) over given wavelengths."""
    if len(materials) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]

    n_layers = len(materials)
    n_wl = len(wavelengths)

    n_arr = np.array([mat_dict[m] for m in materials], dtype=float)
    d_arr = np.array(thicknesses, dtype=float)

    # Build total characteristic matrix
    M = np.tile(np.eye(2, dtype=complex), (n_wl, 1, 1))

    for n, d in zip(n_arr, d_arr):
        if d == 0.0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        L = np.empty((n_wl, 2, 2), dtype=complex)
        L[:, 0, 0] = cos_b
        L[:, 0, 1] = 1.0j * sin_b / n
        L[:, 1, 0] = n * 1.0j * sin_b
        L[:, 1, 1] = cos_b
        M = M @ L

    Y = M[:, 1, 1] / M[:, 0, 1]
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2
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


def _coordinate_descent(materials, thicknesses, problem, deadline, wavelengths, max_iter=500):
    """Fast coordinate descent with golden-section line search on each coordinate."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses

    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    thicks = np.array(thicknesses, dtype=float)
    best_score = _mean_reflectance_fast(materials, thicks, problem, wavelengths)

    # Use a subset of wavelengths for fast evaluation during search
    n_wl = len(wavelengths)
    if n_wl > 101:
        step_idx = max(1, n_wl // 101)
        fast_wl = wavelengths[::step_idx]
    else:
        fast_wl = wavelengths

    def fast_eval(t):
        if not _is_valid(materials, list(t), problem):
            return best_score + 50.0
        return _mean_reflectance_fast(materials, t, problem, fast_wl)

    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        improved = False
        for i in range(n_layers):
            if time.time() > deadline:
                break

            t = thicks.copy()
            lo = max(0.0, t[i])
            hi = min(max_layer_nm, t[i])

            # Also respect total constraint
            other_sum = np.sum(thicks) - thicks[i]
            hi = min(hi, max_total_nm - other_sum)
            if hi <= lo:
                continue

            # Golden section search
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a = lo
            b = hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)

            fc = fast_eval(np.where(np.arange(n_layers) == i, c, t))
            fd = fast_eval(np.where(np.arange(n_layers) == i, d, t))

            for _ in range(20):
                if time.time() > deadline:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = fast_eval(np.where(np.arange(n_layers) == i, c, t))
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = fast_eval(np.where(np.arange(n_layers) == i, d, t))

            best_new = c if fc < fd else d
            best_new_score = fc if fc < fd else fd

            if best_new_score < fast_eval(thicks) - 1e-8:
                thicks[i] = best_new
                improved = True

        if not improved:
            break

    # Final evaluation on full grid
    final_score = _mean_reflectance_fast(materials, thicks, problem, wavelengths)
    if final_score < best_score:
        best_score = final_score

    return materials, list(thicks)


def design(problem):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

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

    # Quarter-wave at center
    dL_qw = centre / (4.0 * nL)
    dH_qw = centre / (4.0 * nH)

    candidates = []

    # Template 1: Alternating L/H starting with L (odd layers)
    for n in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        if n > max_layers:
            break
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        thicks = [dL_qw if i % 2 == 0 else dH_qw for i in range(n)]
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Template 2: Alternating L/H starting with L (even layers)
    for n in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n > max_layers:
            break
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        thicks = [dL_qw if i % 2 == 0 else dH_qw for i in range(n)]
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Template 3: Symmetric gradient - thicknesses vary geometrically
    # For broadband AR, layers should have varying optical thicknesses
    for n in [5, 7, 9, 11, 13, 15]:
        if n > max_layers:
            continue
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        # Geometric thickness variation: thinner in middle, thicker at edges
        # or vice versa - try both
        for factor in [0.7, 0.8, 1.0, 1.2]:
            thicks = []
            for i in range(n):
                base = dL_qw if i % 2 == 0 else dH_qw
                # Taper: layers near edges get thicker
                pos = i / max(1, n - 1)
                taper = 1.0 + (factor - 1.0) * (1.0 - abs(pos - 0.5) * 2.0)
                t = base * taper
                thicks.append(min(max(t, 0.0), max_layer_nm))
            if sum(thicks) <= max_total_nm:
                candidates.append((mats, thicks))

    # Template 4: All-L then all-H (step index profile)
    for nL_count in [2, 3, 4, 5, 6]:
        for nH_count in [1, 2, 3, 4]:
            n = nL_count + nH_count
            if n > max_layers:
                continue
            mats = ["L"] * nL_count + ["H"] * nH_count
            thicks = [dL_qw] * nL_count + [dH_qw] * nH_count
            if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
                candidates.append((mats, thicks))

    # Template 5: H-first alternating
    for n in [3, 5, 7, 9, 11]:
        if n > max_layers:
            continue
        mats = ["H" if i % 2 == 0 else "L" for i in range(n)]
        thicks = [dH_qw if i % 2 == 0 else dL_qw for i in range(n)]
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Template 6: Optimized optical thicknesses - use different center wavelengths
    for wl_c in [450, 500, 550, 600, 650]:
        dL_c = wl_c / (4.0 * nL)
        dH_c = wl_c / (4.0 * nH)
        for n in [5, 7, 9, 11, 13]:
            if n > max_layers:
                continue
            mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
            thicks = [dL_c if i % 2 == 0 else dH_c for i in range(n)]
            if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
                candidates.append((mats, thicks))

    # Template 7: Very thick outer layers (broadband trick)
    for n in [3, 5, 7]:
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        thicks = []
        for i in range(n):
            if i == 0 or i == n - 1:
                base = dL_qw if i % 2 == 0 else dH_qw
                thicks.append(min(base * 1.5, max_layer_nm))
            else:
                base = dL_qw if i % 2 == 0 else dH_qw
                thicks.append(base)
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Deduplicate
    seen = set()
    unique_candidates = []
    for mats, thicks in candidates:
        key = tuple(mats) + tuple(round(t, 2) for t in thicks)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((mats, thicks))

    # Evaluate all candidates on full grid
    scored = []
    for mats, thicks in unique_candidates:
        if time.time() > deadline - 20.0:
            break
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_fast(mats, thicks, problem, wavelengths)
        scored.append((score, mats, thicks))

    scored.sort(key=lambda x: x[0])

    if not scored:
        # Fallback: single quarter-wave L layer
        mats = ["L"]
        thicks = [dL_qw]
        return mats, thicks

    best_score, best_mats, best_thicks = scored[0]

    # Local search on top candidates
    n_to_refine = min(8, len(scored))
    for i in range(n_to_refine):
        if time.time() > deadline - 5.0:
            break
        mats_c, thicks_c = scored[i][1], scored[i][2]
        refined_mats, refined_thicks = _coordinate_descent(
            mats_c, thicks_c, problem, deadline - 3.0, wavelengths
        )
        refined_score = _mean_reflectance_fast(refined_mats, refined_thicks, problem, wavelengths)
        if refined_score < best_score:
            best_score = refined_score
            best_mats = refined_mats
            best_thicks = refined_thicks

    # Ensure validity of final result
    if not _is_valid(best_mats, best_thicks, problem):
        # Fallback to first valid candidate
        best_mats, best_thicks = scored[0][1], scored[0][2]

    return best_mats, best_thicks

# EVOLVE-BLOCK-END
