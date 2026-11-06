"""Broadband antireflection coating design: fast CM + DE local search + structure search."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _mean_reflectance_vec(materials, thicknesses, problem, wavelengths=None):
    """Vectorized mean reflectance (percent) over the band via characteristic-matrix method.

    Uses per-layer complex-exponential matrices to avoid separate cos/sin and matmul cost.
    """
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

    n_arr = np.array([mat_dict[m] for m in materials], dtype=float)
    d_arr = np.array(thicknesses, dtype=float)

    # Total characteristic matrix M (n_wl,2,2)
    M0 = np.ones(n_wl)
    M1 = np.zeros(n_wl, dtype=complex)
    M2 = np.zeros(n_wl, dtype=complex)
    M3 = np.ones(n_wl)

    for n, d in zip(n_arr, d_arr):
        if d == 0:
            continue
        # exp(i beta) and exp(-i beta)
        beta = 2.0 * np.pi * n * d / wavelengths
        e_plus = np.exp(1j * beta)
        e_minus = np.exp(-1j * beta)
        cos_b = 0.5 * (e_plus + e_minus)
        sin_b = 0.5j * (e_plus - e_minus)
        # Layer matrix elements
        L00 = cos_b
        L01 = 1j * sin_b / n
        L10 = n * 1j * sin_b
        L11 = cos_b
        # M = M @ L
        nM0 = M0 * L00 + M1 * L10
        nM1 = M0 * L01 + M1 * L11
        nM2 = M2 * L00 + M3 * L10
        nM3 = M2 * L01 + M3 * L11
        M0, M1, M2, M3 = nM0, nM1, nM2, nM3

    # Substrate admittance Ys = ns
    Y = M3 / M1
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


def _clamp_thicknesses(v, max_layer_nm, max_total_nm):
    """Clip a thickness vector to per-layer and total constraints."""
    v = np.clip(v, 0.0, max_layer_nm)
    s = v.sum()
    if s > max_total_nm and s > 0:
        v = v * (max_total_nm / s)
    return v


def _local_search(materials, thicknesses, problem, deadline, wavelengths, max_iter=80):
    """Differential-evolution style local search on thicknesses."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses

    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    best_thicks = _clamp_thicknesses(np.array(thicknesses, dtype=float), max_layer_nm, max_total_nm)
    best_score = _mean_reflectance_vec(materials, best_thicks, problem, wavelengths)

    # Population size
    pop_size = min(24, max(8, 2 * n_layers + 4))
    step = 15.0  # initial step size (nm)

    # Initialize population: best + random perturbations
    pop = [best_thicks.copy()]
    for _ in range(pop_size - 1):
        v = best_thicks + np.random.randn(n_layers) * step
        v = _clamp_thicknesses(v, max_layer_nm, max_total_nm)
        pop.append(v)

    scores = []
    for v in pop:
        if not _is_valid(materials, list(v), problem):
            scores.append(best_score + 100.0)
        else:
            scores.append(_mean_reflectance_vec(materials, list(v), problem, wavelengths))

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Update best
        for i, sc in enumerate(scores):
            if sc < best_score:
                best_score = sc
                best_thicks = pop[i].copy()

        # Adaptive step: shrink if no improvement recently
        if it > 0 and it % 10 == 0:
            # Check if population is converging
            if max(scores) - min(scores) < 1e-4:
                step *= 0.7

        # Generate new candidates via DE: x_new = x_rand1 + F*(x_rand2 - x_rand3)
        F = 0.8
        new_candidates = []
        for _ in range(pop_size):
            i1, i2, i3 = np.random.choice(pop_size, size=3, replace=False)
            v = pop[i1] + F * (pop[i2] - pop[i3])
            v = _clamp_thicknesses(v, max_layer_nm, max_total_nm)
            new_candidates.append(v)

        # Evaluate new candidates
        new_scores = []
        for v in new_candidates:
            if not _is_valid(materials, list(v), problem):
                new_scores.append(best_score + 100.0)
            else:
                new_scores.append(_mean_reflectance_vec(materials, list(v), problem, wavelengths))

        # Selection: keep better of old and new for each slot
        for i in range(pop_size):
            if new_scores[i] < scores[i]:
                pop[i] = new_candidates[i]
                scores[i] = new_scores[i]

    return materials, list(best_thicks)


def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s wall-clock limit

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

    rng = np.random.RandomState(42)

    best_score = float('inf')
    best_result = None

    # --- Phase 1: Generate diverse candidate structures and evaluate ---
    candidates = []

    # Systematic alternating stacks (both starting materials, both parities)
    for n_layers in range(1, min(max_layers, 20) + 1):
        for start_mat in ['L', 'H']:
            mats = []
            thicks = []
            for i in range(n_layers):
                m = start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'L')
                mats.append(m)
                thicks.append(dL if m == 'L' else dH)
            if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
                candidates.append((mats, thicks))

    # Random structures for diversity
    for _ in range(30):
        n_layers = rng.randint(3, min(max_layers, 15) + 1)
        mats = [rng.choice(['L', 'H']) for _ in range(n_layers)]
        thicks = [dL if m == 'L' else dH for m in mats]
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Evaluate all candidates
    scored_candidates = []
    for mats, thicks in candidates:
        if time.time() > deadline - 60.0:
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

    scored_candidates.sort(key=lambda x: x[0])

    # --- Phase 2: Local search on top candidates ---
    top_k = min(8, len(scored_candidates))
    for score, mats, thicks in scored_candidates[:top_k]:
        if time.time() > deadline - 15.0:
            break
        mats_refined, thicks_refined = _local_search(mats, thicks, problem, deadline - 10.0, wavelengths)
        score_refined = _mean_reflectance_vec(mats_refined, thicks_refined, problem, wavelengths)
        if score_refined < best_score:
            best_score = score_refined
            best_result = (mats_refined, thicks_refined)

    # --- Phase 3: Explore new random structures with local search if time permits ---
    while time.time() < deadline - 20.0:
        n_layers = rng.randint(3, min(max_layers, 16) + 1)
        mats = [rng.choice(['L', 'H']) for _ in range(n_layers)]
        thicks = [dL if m == 'L' else dH for m in mats]
        if sum(thicks) > max_total_nm or any(t > max_layer_nm for t in thicks):
            continue
        if not _is_valid(mats, thicks, problem):
            continue

        mats_refined, thicks_refined = _local_search(mats, thicks, problem, deadline - 15.0, wavelengths)
        score_refined = _mean_reflectance_vec(mats_refined, thicks_refined, problem, wavelengths)
        if score_refined < best_score:
            best_score = score_refined
            best_result = (mats_refined, thicks_refined)

    # Final safety: ensure result is valid
    if best_result is None:
        best_result = (["L"], [dL])

    return best_result

# EVOLVE-BLOCK-END
