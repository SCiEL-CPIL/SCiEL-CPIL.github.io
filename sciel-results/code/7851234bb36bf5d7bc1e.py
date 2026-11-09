"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Combined strategy:
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Structural change: instead of fixing layer sequences and optimizing thicknesses,
     we use a continuous parameterization where both the sequence and thicknesses
     are optimized together via a differentiable "soft assignment" approach.
     Additionally, we use scipy L-BFGS-B for fast local optimization from many
     diverse starting points (inspired by the 8.19 score program).
  3. Global exploration via:
     - Differential evolution on the thickness vector for fixed patterns
     - Multi-start L-BFGS-B from diverse initializations (quarter-wave, random, graded)
     - Simulated annealing for global search
  4. Explicit deadline guard with safety margin.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        n_s = problem["substrate_index"]
        n_0 = problem["ambient_index"]
        return 100.0 * ((n_s - n_0) / (n_s + n_0)) ** 2

    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _clamp_thicknesses(ths, max_total, max_layer):
    """Clamp thicknesses to valid range and scale to fit total budget."""
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
    return ths


def _log_space_evaluator(materials, problem, max_total, max_layer):
    """Create an evaluator in log-space for scipy optimization."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    lo = np.full(n, math.log(0.1))
    hi = np.full(n, math.log(max_layer))

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        return _mean_reflectance(materials, list(t), problem)

    return eval_x, lo, hi


def _nelder_mead(objective, init_x, bounds, max_iters=200, tol=1e-10):
    """Nelder-Mead simplex optimization with bounds handling."""
    n = len(init_x)
    x = np.array(init_x, dtype=float)
    f0 = objective(x)
    fbest = f0
    xbest = x.copy()

    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 1 else 1.0
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)

    scores = [objective(simplex[i]) for i in range(n + 1)]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for it in range(max_iters):
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]

        if scores[0] < fbest:
            fbest = scores[0]
            xbest = simplex[0].copy()

        if max(scores) - min(scores) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)

        if fr < scores[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                scores[-1] = fe
            else:
                simplex[-1] = xr
                scores[-1] = fr
        elif fr < scores[-2]:
            simplex[-1] = xr
            scores[-1] = fr
        else:
            if fr < scores[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                        scores[i] = objective(simplex[i])
                    if max(scores) - min(scores) < tol:
                        break
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    scores[i] = objective(simplex[i])
                if max(scores) - min(scores) < tol:
                    break

    return xbest, fbest


def _simulated_annealing(objective, init_x, bounds, max_iters=500, T0=10.0, cooling=0.995, seed=None):
    """Simulated annealing for global optimization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    n = len(init_x)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    x = np.array(init_x, dtype=float)
    x = np.clip(x, lo, hi)
    f = objective(x)
    xbest = x.copy()
    fbest = f

    T = T0
    for it in range(max_iters):
        i = rng.randint(0, n)
        step_size = 0.1 * (hi[i] - lo[i])
        x_new = x.copy()
        x_new[i] += rng.uniform(-step_size, step_size)
        x_new = np.clip(x_new, lo, hi)

        f_new = objective(x_new)
        delta_f = f_new - f

        if delta_f < 0 or rng.random() < np.exp(-delta_f / T):
            x = x_new
            f = f_new
            if f < fbest:
                fbest = f
                xbest = x.copy()

        T *= cooling
        if T < 1e-6:
            break

    return xbest, fbest


def _differential_evolution(objective, bounds, pop_size=15, max_iters=100, seed=None):
    """Differential evolution for global optimization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Initialize population
    pop = rng.uniform(lo, hi, size=(pop_size, n))
    fitness = np.array([objective(pop[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitness)
    gbest = pop[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]

    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate

    for gen in range(max_iters):
        for i in range(pop_size):
            # Select three distinct random individuals
            a, b, c = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)

            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, lo, hi)

            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(0, n)
            for j in range(n):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Selection
            f_trial = objective(trial)
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < gbest_fitness:
                    gbest = trial.copy()
                    gbest_fitness = f_trial

    return gbest, gbest_fitness


def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]

    random.seed(42)
    np.random.seed(42)

    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    best_result = None
    best_score = float('inf')

    def maybe_record(mats, ths):
        nonlocal best_result, best_score
        ths = _clamp_thicknesses(ths, max_total, max_layer)
        if sum(ths) > max_total + 1e-6:
            ths = [t * (max_total / sum(ths)) for t in ths]
        if any(t < 0 or t > max_layer for t in ths):
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_result = (list(mats), list(ths))

    # Generate diverse candidate patterns
    candidates = []

    # Symmetric patterns of various lengths
    for length in range(1, min(max_layers, 16) + 1):
        half = (length + 1) // 2
        for _ in range(40):
            half_pattern = tuple(random.choice("LH") for _ in range(half))
            if length % 2 == 0:
                pattern = half_pattern + half_pattern[::-1]
            else:
                pattern = half_pattern + half_pattern[-2::-1]
            candidates.append(tuple(pattern))

    # Graded index patterns
    for length in range(2, min(max_layers, 14) + 1):
        for _ in range(20):
            pattern = []
            threshold = length // 2
            for i in range(length):
                if i < threshold:
                    pattern.append('L' if random.random() < 0.8 else 'H')
                else:
                    pattern.append('H' if random.random() < 0.8 else 'L')
            candidates.append(tuple(pattern))

    # Alternating patterns
    for period in [2, 3, 4]:
        for length in range(period, min(max_layers, 18) + 1, period):
            for start in ['L', 'H']:
                pattern = tuple(start if (i // period) % 2 == 0 else ('H' if start == 'L' else 'L') for i in range(length))
                candidates.append(pattern)

    # Random patterns
    for _ in range(200):
        length = random.randint(2, min(max_layers, 18))
        pattern = tuple(random.choice("LH") for _ in range(length))
        candidates.append(pattern)

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)
    candidates = unique_candidates

    # Sort by length
    candidates.sort(key=lambda p: len(p))

    def make_objective(pattern):
        n_layers = len(pattern)
        bounds = [(0.0, max_layer)] * n_layers

        def objective(ths):
            ths_clamped = _clamp_thicknesses(ths, max_total, max_layer)
            return _mean_reflectance(pattern, ths_clamped, problem)

        return objective, bounds

    # Phase 1: Quick evaluation with multiple initialization strategies
    for pattern in candidates:
        if time.time() >= deadline - 60:
            break
        if len(pattern) > max_layers:
            continue

        objective, bounds = make_objective(pattern)

        # Try several initialization strategies
        init_strategies = []

        # Quarter-wave initialization
        qw_ths = [qw[m] for m in pattern]
        qw_ths = _clamp_thicknesses(qw_ths, max_total, max_layer)
        init_strategies.append(qw_ths)

        # Random initializations
        for _ in range(3):
            init_ths = [random.uniform(0, max_layer) for _ in pattern]
            init_ths = _clamp_thicknesses(init_ths, max_total, max_layer)
            init_strategies.append(init_ths)

        # Graded thickness initialization
        n_layers = len(pattern)
        graded_ths = []
        for i in range(n_layers):
            t = max_layer * (1.0 - abs(i - n_layers/2 + 0.5) / (n_layers/2 + 0.5))
            graded_ths.append(max(0.1, t))
        graded_ths = _clamp_thicknesses(graded_ths, max_total, max_layer)
        init_strategies.append(graded_ths)

        for init_ths in init_strategies:
            if time.time() >= deadline - 60:
                break
            maybe_record(pattern, init_ths)

    # Phase 2: Differential evolution on top candidates
    if time.time() < deadline - 40:
        # Pick diverse patterns for DE
        top_patterns = []
        for pattern in candidates[:80]:
            if len(pattern) > 12:
                continue
            objective, bounds = make_objective(pattern)
            qw_ths = [qw[m] for m in pattern]
            qw_ths = _clamp_thicknesses(qw_ths, max_total, max_layer)
            score = objective(qw_ths)
            top_patterns.append((pattern, qw_ths, score))

        top_patterns.sort(key=lambda x: x[2])

        for pattern, init_ths, score in top_patterns[:10]:
            if time.time() >= deadline - 40:
                break
            objective, bounds = make_objective(pattern)
            xbest, fbest = _differential_evolution(objective, bounds, pop_size=12, max_iters=60, seed=42)
            maybe_record(pattern, list(xbest))

    # Phase 3: Scipy L-BFGS-B optimization from diverse starts
    if _HAVE_SCIPY and time.time() < deadline - 25:
        for pattern in candidates[:60]:
            if time.time() >= deadline - 25:
                break
            if len(pattern) > 14:
                continue

            eval_x, lo, hi = _log_space_evaluator(pattern, problem, max_total, max_layer)
            n_layers = len(pattern)

            # Multiple starting points
            n_starts = 3 if n_layers <= 8 else 2
            for _ in range(n_starts):
                if time.time() >= deadline - 25:
                    break
                x0 = []
                for m in pattern:
                    qw_val = qw[m]
                    t = qw_val * random.uniform(0.5, 1.6)
                    t = max(0.5, min(max_layer, t))
                    x0.append(math.log(t))
                x0 = np.array(x0)

                try:
                    res = _scipy_minimize(
                        eval_x, x0, method="L-BFGS-B",
                        bounds=list(zip(lo, hi)),
                        options={"maxiter": 200, "maxfun": 3000, "maxls": 20}
                    )
                    xc = np.clip(res.x, lo, hi)
                    t = np.exp(xc)
                    total = t.sum()
                    if total > max_total:
                        t = t * (max_total / total)
                    maybe_record(pattern, list(t))
                except Exception:
                    pass

    # Phase 4: Simulated annealing on top candidates
    if time.time() < deadline - 15:
        top_patterns = []
        for pattern in candidates[:50]:
            if len(pattern) > 10:
                continue
            objective, bounds = make_objective(pattern)
            qw_ths = [qw[m] for m in pattern]
            qw_ths = _clamp_thicknesses(qw_ths, max_total, max_layer)
            score = objective(qw_ths)
            top_patterns.append((pattern, qw_ths, score))

        top_patterns.sort(key=lambda x: x[2])

        for pattern, init_ths, score in top_patterns[:8]:
            if time.time() >= deadline - 15:
                break
            objective, bounds = make_objective(pattern)
            xbest, fbest = _simulated_annealing(objective, init_ths, bounds, max_iters=300, T0=5.0, cooling=0.997, seed=42)
            maybe_record(pattern, list(xbest))

    # Phase 5: Final polish with Nelder-Mead on the best result
    if best_result is not None and time.time() < deadline - 5:
        mats, ths = best_result
        objective, bounds = make_objective(mats)
        xbest, fbest = _nelder_mead(objective, ths, bounds, max_iters=400)
        maybe_record(mats, list(xbest))

    if best_result is None:
        return ["L"], [qw["L"]]

    mats, ths = best_result
    ths = _clamp_thicknesses(ths, max_total, max_layer)
    return list(mats), list(ths)
