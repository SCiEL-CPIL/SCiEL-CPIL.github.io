"""Broadband antireflection coating design: vectorized CM + multistart local search."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _mean_reflectance_vec(materials, thicknesses, problem, wavelengths=None):
    """Vectorized mean reflectance (percent) over the band using the characteristic-matrix method."""
    if len(materials) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    if wavelengths is None:
        band = problem["band_nm"]
        wavelengths = np.linspace(band[0], band[1], 401)

    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]

    n_layers = len(materials)
    n_wl = len(wavelengths)

    # Precompute per-layer index and thickness arrays
    n_arr = np.array([mat_dict[m] for m in materials], dtype=float)
    d_arr = np.array(thicknesses, dtype=float)

    # Build total characteristic matrix M for all wavelengths simultaneously
    # M has shape (n_wl, 2, 2)
    M = np.tile(np.eye(2, dtype=complex), (n_wl, 1, 1))

    for n, d in zip(n_arr, d_arr):
        if d == 0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        # Layer matrix shape (n_wl, 2, 2)
        L = np.empty((n_wl, 2, 2), dtype=complex)
        L[:, 0, 0] = cos_b
        L[:, 0, 1] = 1.0j * sin_b / n
        L[:, 1, 0] = n * 1.0j * sin_b
        L[:, 1, 1] = cos_b
        M = M @ L

    # Substrate admittance Ys = ns
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


def _local_search(materials, thicknesses, problem, deadline, wavelengths):
    """Nelder-Mead style local search on thicknesses with validity constraints."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses

    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    best_thicks = np.array(thicknesses, dtype=float)
    best_score = _mean_reflectance_vec(materials, best_thicks, problem, wavelengths)

    # Initial simplex: perturb each coordinate by a small fraction
    step = 5.0  # nm
    simplex = [best_thicks.copy()]
    for i in range(n_layers):
        v = best_thicks.copy()
        v[i] += step
        if v[i] < 0:
            v[i] = 0
        if v[i] > max_layer_nm:
            v[i] = max_layer_nm
        if sum(v) > max_total_nm:
            # Scale down proportionally
            if sum(v) > 0:
                scale = max_total_nm / sum(v)
                v = v * scale
        simplex.append(v)

    scores = []
    for v in simplex:
        if not _is_valid(materials, list(v), problem):
            scores.append(best_score + 100.0)
        else:
            scores.append(_mean_reflectance_vec(materials, list(v), problem, wavelengths))

    # Order simplex by score
    order = np.argsort(scores)
    simplex = [simplex[i] for i in order]
    scores = [scores[i] for i in order]

    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    max_iter = 300
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Check convergence
        if max(scores) - min(scores) < 1e-6:
            break

        worst_idx = n_layers  # last element is worst
        centroid = np.mean(simplex[:n_layers], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[worst_idx])
        # Clip to valid range
        xr = np.clip(xr, 0, max_layer_nm)
        if sum(xr) > max_total_nm and sum(xr) > 0:
            xr = xr * (max_total_nm / sum(xr))

        if not _is_valid(materials, list(xr), problem):
            xr_score = best_score + 100.0
        else:
            xr_score = _mean_reflectance_vec(materials, list(xr), problem, wavelengths)

        if xr_score < scores[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, 0, max_layer_nm)
            if sum(xe) > max_total_nm and sum(xe) > 0:
                xe = xe * (max_total_nm / sum(xe))
            if not _is_valid(materials, list(xe), problem):
                xe_score = best_score + 100.0
            else:
                xe_score = _mean_reflectance_vec(materials, list(xe), problem, wavelengths)

            if xe_score < xr_score:
                simplex[worst_idx] = xe
                scores[worst_idx] = xe_score
            else:
                simplex[worst_idx] = xr
                scores[worst_idx] = xr_score
        elif xr_score < scores[n_layers - 1]:
            simplex[worst_idx] = xr
            scores[worst_idx] = xr_score
        else:
            # Contraction
            if xr_score < scores[n_layers - 1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (xr - centroid)
            xc = np.clip(xc, 0, max_layer_nm)
            if sum(xc) > max_total_nm and sum(xc) > 0:
                xc = xc * (max_total_nm / sum(xc))

            if not _is_valid(materials, list(xc), problem):
                xc_score = best_score + 100.0
            else:
                xc_score = _mean_reflectance_vec(materials, list(xc), problem, wavelengths)

            if xc_score < scores[n_layers - 1]:
                simplex[worst_idx] = xc
                scores[worst_idx] = xc_score
            else:
                # Shrink
                for i in range(1, n_layers + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], 0, max_layer_nm)
                    if sum(simplex[i]) > max_total_nm and sum(simplex[i]) > 0:
                        simplex[i] = simplex[i] * (max_total_nm / sum(simplex[i]))
                    if not _is_valid(materials, list(simplex[i]), problem):
                        scores[i] = best_score + 100.0
                    else:
                        scores[i] = _mean_reflectance_vec(materials, list(simplex[i]), problem, wavelengths)

        # Reorder
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]

        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_thicks = simplex[0].copy()

    return materials, list(best_thicks)


def design(problem):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s wall-clock limit

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

    # Quarter-wave thicknesses at center wavelength
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)

    candidates = []

    # Symmetric stacks starting and ending with L
    for n_layers in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        if n_layers > max_layers:
            break
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("L")
                thicks.append(dL)
            else:
                mats.append("H")
                thicks.append(dH)
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Stacks starting with L ending with H
    for n_layers in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            break
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("L")
                thicks.append(dL)
            else:
                mats.append("H")
                thicks.append(dH)
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Stacks starting with H ending with L
    for n_layers in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            break
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("H")
                thicks.append(dH)
            else:
                mats.append("L")
                thicks.append(dL)
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Stacks starting and ending with H
    for n_layers in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        if n_layers > max_layers:
            break
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("H")
                thicks.append(dH)
            else:
                mats.append("L")
                thicks.append(dL)
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Try with slightly different center wavelengths
    for wl_center in [500, 550, 600]:
        dL_c = wl_center / (4.0 * nL)
        dH_c = wl_center / (4.0 * nH)
        for n_layers in [3, 5, 7, 9]:
            if n_layers > max_layers:
                continue
            # L-H-L pattern
            mats = []
            thicks = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mats.append("L")
                    thicks.append(dL_c)
                else:
                    mats.append("H")
                    thicks.append(dH_c)
            if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
                candidates.append((mats, thicks))

    # Evaluate all candidates
    best_score = float('inf')
    best_result = None
    scored_candidates = []

    for mats, thicks in candidates:
        if time.time() > deadline - 30.0:
            break
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_vec(mats, thicks, problem, wavelengths)
        scored_candidates.append((score, mats, thicks))
        if score < best_score:
            best_score = score
            best_result = (mats, list(thicks))

    if best_result is None:
        best_result = (["L"], [dL])
        best_score = _mean_reflectance_vec(best_result[0], best_result[1], problem, wavelengths)

    # Sort candidates by score
    scored_candidates.sort(key=lambda x: x[0])

    # Local search on top candidates
    for score, mats, thicks in scored_candidates[:5]:
        if time.time() > deadline - 10.0:
            break
        mats_refined, thicks_refined = _local_search(mats, thicks, problem, deadline - 5.0, wavelengths)
        score_refined = _mean_reflectance_vec(mats_refined, thicks_refined, problem, wavelengths)
        if score_refined < best_score:
            best_score = score_refined
            best_result = (mats_refined, thicks_refined)

    return best_result

# EVOLVE-BLOCK-END
