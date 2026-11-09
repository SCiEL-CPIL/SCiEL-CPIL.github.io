"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRUCTURAL CHANGE: Differential Evolution (population-based global optimizer)
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Fixed 20-layer alternating L/H structure (graded-index approximation).
  3. Differential Evolution for global exploration of thickness landscape,
     followed by L-BFGS-B local refinement from the best DE population member.
  4. Explicit deadline guard: return the best feasible result found before the limit.
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
    """Mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
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


def _mean_reflectance_fast(ns, ds, wavs, n_0, n_s):
    """Fast version with precomputed wavelengths and index array."""
    n_ds = ns * ds
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(ns)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _make_evaluator(materials, problem, max_total, max_layer):
    """Return (eval, lo, hi). eval(x) takes a log-scaled length vector x (length n)
    and returns the mean reflectance for the corresponding physical thicknesses,
    where t_j = clip(exp(x_j), 0, max_layer) and the whole stack is scaled to satisfy
    sum(t) <= max_total."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    lo = np.full(n, math.log(1.0))
    hi = np.full(n, math.log(max_layer))
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        return _mean_reflectance_fast(ns, t, wavs, n_0, n_s)

    return eval_x, lo, hi


def _nm_fallback(f, x0, lo, hi, deadline):
    """Lightweight Nelder-Mead (constrained to box + total) as a scipy fallback."""
    n = len(x0)
    best = list(x0)
    best_val = f(best)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] = hi[i] if pt[i] == lo[i] else lo[i] + 0.2 * (hi[i] - lo[i])
        simplex.append(pt)
    scores = [f(pt) for pt in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    while it < 800 and time.time() < deadline:
        it += 1
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        if scores[0] < best_val:
            best_val = scores[0]
            best = list(simplex[0])
        centroid = np.mean(simplex[:-1], axis=0)
        xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        fr = f(list(xr))
        if fr < scores[0]:
            xe = np.clip(centroid + gamma * (centroid - simplex[-1]), lo, hi)
            fe = f(list(xe))
            if fe < fr:
                simplex[-1], scores[-1] = list(xe), fe
            else:
                simplex[-1], scores[-1] = list(xr), fr
        elif fr < scores[-2]:
            simplex[-1], scores[-1] = list(xr), fr
        else:
            xc = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
            fc = f(list(xc))
            if fc <= fr:
                simplex[-1], scores[-1] = list(xc), fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = np.clip(
                        np.array(simplex[0]) + sigma * (np.array(simplex[i]) - np.array(simplex[0])),
                        lo, hi)
                    scores[i] = f(list(simplex[i]))
                if max(scores) - min(scores) < 1e-9:
                    break
    return best, best_val


def design(problem):
    start_time = time.time()
    deadline = start_time + 120  # safety margin before the 150 s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(2024)
    np.random.seed(2024)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    qw_L = wl_center / (4.0 * nL)
    qw_H = wl_center / (4.0 * nH)

    best_ths = None
    best_mats = None
    best_score = float("inf")

    def maybe_record(mats, ths):
        nonlocal best_ths, best_mats, best_score
        total = sum(ths)
        if total > max_total + 1e-6:
            ths = [t * (max_total / total) for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        if any(t < 0 or t > max_layer for t in ths):
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)

    # Fixed 20-layer alternating L/H structure (graded-index approximation)
    n_layers = min(20, max_layers)
    materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    eval_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)
    
    # Quarter-wave reference for seeding
    qw_ref = np.array([qw_L if m == 'L' else qw_H for m in materials])
    
    # Phase 1: Differential Evolution for global exploration
    if _HAVE_SCIPY and time.time() < deadline - 30:
        try:
            from scipy.optimize import differential_evolution as _de
        except Exception:
            _de = None
        
        if _de is not None:
            # DE bounds in log-space
            bounds = list(zip(lo, hi))
            
            # Seed DE with quarter-wave solution as one of the initial population
            # DE will generate its own population, but we can use init='differential'
            # which creates a diverse population around the midpoint
            
            # Use a moderate population size for good exploration
            popsize = 15
            maxiter_de = 50  # conservative to stay within time budget
            
            # Time-limited DE: run and check time after each generation
            # We'll do a manual loop with time checking
            n = len(lo)
            
            # Initialize population: one member at quarter-wave, rest random
            pop = np.zeros((popsize, n))
            pop[0] = np.log(np.clip(qw_ref, 1.0, max_layer))
            for i in range(1, popsize):
                for j in range(n):
                    pop[i, j] = random.uniform(lo[j], hi[j])
            
            # Evaluate initial population
            pop_scores = np.array([eval_x(pop[i]) for i in range(popsize)])
            best_idx = np.argmin(pop_scores)
            
            # DE parameters
            F = 0.8  # differential weight
            CR = 0.9  # crossover rate
            
            for gen in range(maxiter_de):
                if time.time() >= deadline - 30:
                    break
                
                # Sort population by fitness
                order = np.argsort(pop_scores)
                pop = pop[order]
                pop_scores = pop_scores[order]
                
                # Update best
                t_best = np.exp(pop[0])
                total = t_best.sum()
                if total > max_total:
                    t_best = t_best * (max_total / total)
                maybe_record(materials, list(t_best))
                
                # Generate new population
                new_pop = np.copy(pop)
                new_scores = np.copy(pop_scores)
                
                for i in range(popsize):
                    # Differential evolution mutation
                    # Pick 3 distinct random indices different from i
                    while True:
                        a, b, c = random.sample(range(popsize), 3)
                        if a != i and b != i and c != i:
                            break
                    
                    # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                    v = pop[a] + F * (pop[b] - pop[c])
                    v = np.clip(v, lo, hi)
                    
                    # Binomial crossover
                    trial = np.copy(pop[i])
                    for j in range(n):
                        if random.random() < CR or j == random.randint(0, n-1):
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][j]
                    trial = np.clip(trial, lo, hi)
                    
                    # Evaluate trial
                    trial_score = eval_x(trial)
                    
                    # Selection
                    if trial_score <= pop_scores[i]:
                        new_pop[i] = trial
                        new_scores[i] = trial_score
                
                pop = new_pop
                pop_scores = new_scores
            
            # After DE, take the best member and refine with L-BFGS-B
            order = np.argsort(pop_scores)
            best_x = pop[order[0]]
            best_x = np.clip(best_x, lo, hi)
            
            # Phase 2: L-BFGS-B refinement from DE best
            if time.time() < deadline - 10:
                res = _scipy_minimize(
                    eval_x, best_x, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "maxfun": 8000, "maxls": 50,
                             "ftol": 1e-12, "gtol": 1e-10},
                )
                xc = np.clip(res.x, lo, hi)
                t = np.exp(xc)
                total = t.sum()
                if total > max_total:
                    t = t * (max_total / total)
                maybe_record(materials, list(t))
            
            # Also try refining a few other good DE members
            for idx in order[1:4]:
                if time.time() >= deadline - 5:
                    break
                x0 = pop[idx]
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 300, "maxfun": 5000, "maxls": 40},
                )
                xc = np.clip(res.x, lo, hi)
                t = np.exp(xc)
                total = t.sum()
                if total > max_total:
                    t = t * (max_total / total)
                maybe_record(materials, list(t))
        else:
            # Fallback to original multi-start if DE not available
            _run_fallback_multistart(eval_x, lo, hi, materials, qw_L, qw_H,
                                     max_layer, max_total, maybe_record, deadline)
    else:
        # Fallback to original multi-start if no scipy or not enough time
        _run_fallback_multistart(eval_x, lo, hi, materials, qw_L, qw_H,
                                 max_layer, max_total, maybe_record, deadline)

    # Final polish with Nelder-Mead if time allows
    if best_ths is not None and time.time() < deadline - 4:
        eval_x2, lo2, hi2 = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0_nm = [math.log(max(1.0, t)) for t in best_ths]
        xc_nm, val_nm = _nm_fallback(eval_x2, x0_nm, lo2, hi2, deadline - 2)
        t_nm = np.exp(np.clip(xc_nm, lo2, hi2))
        total_nm = t_nm.sum()
        if total_nm > max_total:
            t_nm = t_nm * (max_total / total_nm)
        maybe_record(best_mats, list(t_nm))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center.
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)


def _run_fallback_multistart(eval_x, lo, hi, materials, qw_L, qw_H,
                              max_layer, max_total, maybe_record, deadline):
    """Fallback multi-start L-BFGS-B if DE is not available."""
    n = len(lo)
    n_starts = 8
    starts = []
    
    # Quarter-wave start
    qw_ref = np.array([qw_L if m == 'L' else qw_H for m in materials])
    x0_qw = np.log(np.clip(qw_ref, 1.0, max_layer))
    starts.append(x0_qw)
    
    # Random starts
    for _ in range(n_starts - 1):
        x0 = np.array([random.uniform(lo[i], hi[i]) for i in range(n)])
        starts.append(x0)
    
    for x0 in starts:
        if time.time() >= deadline - 10:
            break
        if _HAVE_SCIPY:
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 300, "maxfun": 4000, "maxls": 30},
            )
            xc = np.clip(res.x, lo, hi)
        else:
            xc, _ = _nm_fallback(eval_x, list(x0), lo, hi, deadline - 10)
        t = np.exp(xc)
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        maybe_record(materials, list(t))
