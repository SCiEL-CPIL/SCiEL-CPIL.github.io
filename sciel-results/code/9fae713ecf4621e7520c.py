"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Fast exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Systematic enumeration of symmetric patterns (known optimal for broadband AR)
     plus diverse random patterns for coverage.
  3. For each pattern, optimize thicknesses using:
     - Multiple random restarts with quarter-wave initialization
     - Nelder-Mead local optimization (primary refinement)
     - Differential evolution for global search on top candidates
     - Simulated annealing for additional exploration
  4. Keep the best result found within the time budget.
"""

import time
import random
import numpy as np


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

    A = np.ones_like(wavs, dtype=complex)
    B = np.zeros_like(wavs, dtype=complex)
    C = np.zeros_like(wavs, dtype=complex)
    D = np.ones_like(wavs, dtype=complex)

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
        # Second pass: if still over, reduce proportionally again
        total2 = sum(ths)
        if total2 > max_total:
            scale2 = max_total / total2
            ths = [t * scale2 for t in ths]
            ths = [max(0.0, min(max_layer, t)) for t in ths]
    return ths


def _nelder_mead(objective, init_x, bounds, max_iters=300, tol=1e-9, deadline=None):
    """Nelder-Mead simplex optimization with bounds handling."""
    n = len(init_x)
    x = np.array(init_x, dtype=float)
    x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
    f0 = objective(x)
    fbest = f0
    xbest = x.copy()

    # Build simplex with adaptive step size
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.2 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 1 else 1.0
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)

    scores = [objective(simplex[i]) for i in range(n + 1)]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

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
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                        scores[i] = objective(simplex[i])
                    if max(scores) - min(scores) < tol:
                        break
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    scores[i] = objective(simplex[i])
                if max(scores) - min(scores) < tol:
                    break

    return xbest, fbest


def _simulated_annealing(objective, init_x, bounds, max_iters=400, T0=8.0, cooling=0.996, seed=None, deadline=None):
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
        if deadline is not None and time.time() > deadline:
            break

        # Generate neighbor: perturb 1-2 random dimensions
        n_perturb = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(n, size=min(n_perturb, n), replace=False)
        x_new = x.copy()
        for idx in indices:
            step_size = 0.15 * (hi[idx] - lo[idx])
            x_new[idx] += rng.uniform(-step_size, step_size)
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
        if T < 1e-7:
            break

    return xbest, fbest


def _differential_evolution(objective, bounds, pop_size=20, max_iters=50, F=0.8, CR=0.7, deadline=None):
    """Differential evolution for global search."""
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fitness = np.array([objective(pop[i]) for i in range(pop_size)])
    best_idx = np.argmin(fitness)
    xbest = pop[best_idx].copy()
    fbest = fitness[best_idx]

    for gen in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        for i in range(pop_size):
            if deadline is not None and time.time() > deadline:
                break

            # Select three distinct random individuals
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)

            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)

            # Crossover
            trial = pop[i].copy()
            for j in range(n):
                if np.random.random() < CR:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
            trial = np.clip(trial, lo, hi)

            # Selection
            ft = objective(trial)
            if ft < fitness[i]:
                pop[i] = trial
                fitness[i] = ft
                if ft < fbest:
                    fbest = ft
                    xbest = trial.copy()

    return xbest, fbest


def _generate_symmetric_patterns(max_len=8):
    """Generate all symmetric patterns up to max_len layers."""
    patterns = []
    
    # 1-layer
    patterns.append(('L',))
    patterns.append(('H',))
    
    # 2-layer: AB (not symmetric in the usual sense, but include)
    for a in ('L', 'H'):
        for b in ('L', 'H'):
            if a != b:
                patterns.append((a, b))
    
    # 3-layer symmetric: ABA
    for a in ('L', 'H'):
        for b in ('L', 'H'):
            if a != b:
                patterns.append((a, b, a))
    
    # 4-layer symmetric: ABBA
    for a in ('L', 'H'):
        for b in ('L', 'H'):
            if a != b:
                patterns.append((a, b, b, a))
    
    # 5-layer symmetric: ABCBA
    for a in ('L', 'H'):
        for b in ('L', 'H'):
            for c in ('L', 'H'):
                if a != c:  # Ensure non-trivial
                    patterns.append((a, b, c, b, a))
    
    # 6-layer symmetric: ABCCBA
    for a in ('L', 'H'):
        for b in ('L', 'H'):
            for c in ('L', 'H'):
                if a != c:
                    patterns.append((a, b, c, c, b, a))
    
    # 7-layer symmetric: ABCDCBA
    for a in ('L', 'H'):
        for b in ('L', 'H'):
            for c in ('L', 'H'):
                for d in ('L', 'H'):
                    if a != d:
                        patterns.append((a, b, c, d, c, b, a))
    
    # 8-layer symmetric: ABCDDCBA
    for a in ('L', 'H'):
        for b in ('L', 'H'):
            for c in ('L', 'H'):
                for d in ('L', 'H'):
                    if a != d:
                        patterns.append((a, b, c, d, d, c, b, a))
    
    return patterns


def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]

    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    random.seed(42)
    np.random.seed(42)

    # Generate candidate patterns: systematic symmetric + diverse random
    candidates = []
    
    # Systematic symmetric patterns (highest priority)
    sym_patterns = _generate_symmetric_patterns(max_len=8)
    candidates.extend(sym_patterns)
    
    # Additional symmetric patterns up to 10 layers
    for length in [9, 10]:
        half = (length + 1) // 2
        for _ in range(20):
            half_pattern = tuple(random.choice("LH") for _ in range(half))
            if length % 2 == 0:
                pattern = half_pattern + half_pattern[::-1]
            else:
                pattern = half_pattern + half_pattern[-2::-1]
            candidates.append(pattern)
    
    # Graded index patterns (smooth transition)
    for length in range(3, min(max_layers, 12) + 1):
        for _ in range(10):
            pattern = []
            threshold = length // 2
            for i in range(length):
                if i < threshold:
                    pattern.append('L' if random.random() < 0.85 else 'H')
                else:
                    pattern.append('H' if random.random() < 0.85 else 'L')
            candidates.append(tuple(pattern))
    
    # Alternating patterns with varying periods
    for period in [2, 3, 4]:
        for length in range(period, min(max_layers, 16) + 1, period):
            for start in ['L', 'H']:
                pattern = tuple(start if (i // period) % 2 == 0 else ('H' if start == 'L' else 'L') for i in range(length))
                candidates.append(pattern)
    
    # Random patterns
    for _ in range(100):
        length = random.randint(2, min(max_layers, 14))
        pattern = tuple(random.choice("LH") for _ in range(length))
        candidates.append(pattern)
    
    # Remove duplicates and sort by length
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen and len(p) <= max_layers:
            seen.add(p)
            unique_candidates.append(p)
    candidates = unique_candidates
    
    # Sort: symmetric patterns first (shorter ones), then others by length
    def pattern_priority(p):
        # Symmetric patterns get priority
        is_sym = (p == p[::-1]) or (p == p[:len(p)//2] + p[:len(p)//2][::-1])
        return (0 if is_sym else 1, len(p))
    
    candidates.sort(key=pattern_priority)

    best_result = None
    best_score = float('inf')

    def make_objective(pattern):
        n_layers = len(pattern)
        bounds = [(0.0, max_layer)] * n_layers

        def objective(ths):
            ths_clamped = _clamp_thicknesses(ths, max_total, max_layer)
            return _mean_reflectance(pattern, ths_clamped, problem)

        return objective, bounds

    # Phase 1: Quick evaluation with multiple initialization strategies for all patterns
    phase1_deadline = deadline - 80
    for pattern in candidates:
        if time.time() >= phase1_deadline:
            break

        objective, bounds = make_objective(pattern)

        # Try several initialization strategies
        init_strategies = []
        
        # Quarter-wave initialization
        qw_ths = [qw[m] for m in pattern]
        qw_ths = _clamp_thicknesses(qw_ths, max_total, max_layer)
        init_strategies.append(qw_ths)
        
        # Random initializations
        for _ in range(2):
            init_ths = [random.uniform(0.2 * max_layer, 0.8 * max_layer) for _ in pattern]
            init_ths = _clamp_thicknesses(init_ths, max_total, max_layer)
            init_strategies.append(init_ths)
        
        # Graded thickness initialization (thicker in middle)
        n_layers = len(pattern)
        graded_ths = []
        for i in range(n_layers):
            t = max_layer * 0.8 * (1.0 - abs(i - n_layers/2 + 0.5) / (n_layers/2 + 0.5))
            graded_ths.append(max(1.0, t))
        graded_ths = _clamp_thicknesses(graded_ths, max_total, max_layer)
        init_strategies.append(graded_ths)

        for init_ths in init_strategies:
            if time.time() >= phase1_deadline:
                break
            score = objective(init_ths)
            if score < best_score:
                best_score = score
                best_result = (list(pattern), list(init_ths))

    # Phase 2: Nelder-Mead on top candidates
    phase2_deadline = deadline - 40
    if time.time() < phase2_deadline:
        # Pick top patterns for NM
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

        for pattern, init_ths, score in top_patterns[:15]:
            if time.time() >= phase2_deadline:
                break
            objective, bounds = make_objective(pattern)
            xbest, fbest = _nelder_mead(objective, init_ths, bounds, max_iters=400, deadline=phase2_deadline)
            fbest = objective(xbest)
            if fbest < best_score:
                best_score = fbest
                best_result = (list(pattern), list(xbest))

    # Phase 3: Differential evolution on top candidates
    if time.time() < deadline - 25:
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
            if time.time() >= deadline - 25:
                break
            objective, bounds = make_objective(pattern)
            xbest, fbest = _differential_evolution(objective, bounds, pop_size=15, max_iters=40, F=0.8, CR=0.7, deadline=deadline - 25)
            fbest = objective(xbest)
            if fbest < best_score:
                best_score = fbest
                best_result = (list(pattern), list(xbest))

    # Phase 4: Simulated annealing on top candidates
    if time.time() < deadline - 15:
        top_patterns = []
        for pattern in candidates[:40]:
            if len(pattern) > 10:
                continue
            objective, bounds = make_objective(pattern)
            qw_ths = [qw[m] for m in pattern]
            qw_ths = _clamp_thicknesses(qw_ths, max_total, max_layer)
            score = objective(qw_ths)
            top_patterns.append((pattern, qw_ths, score))

        top_patterns.sort(key=lambda x: x[2])

        for pattern, init_ths, score in top_patterns[:6]:
            if time.time() >= deadline - 15:
                break
            objective, bounds = make_objective(pattern)
            xbest, fbest = _simulated_annealing(objective, init_ths, bounds, max_iters=300, T0=8.0, cooling=0.996, seed=42, deadline=deadline - 15)
            fbest = objective(xbest)
            if fbest < best_score:
                best_score = fbest
                best_result = (list(pattern), list(xbest))

    # Phase 5: Final polish with Nelder-Mead on the best result
    if best_result is not None and time.time() < deadline - 5:
        mats, ths = best_result
        objective, bounds = make_objective(mats)
        xbest, fbest = _nelder_mead(objective, ths, bounds, max_iters=500, deadline=deadline - 5)
        fbest = objective(xbest)
        if fbest < best_score:
            best_score = fbest
            best_result = (list(mats), list(xbest))

    if best_result is None:
        return ["L"], [qw["L"]]

    mats, ths = best_result
    # Final validation
    ths = _clamp_thicknesses(ths, max_total, max_layer)
    return list(mats), list(ths)
