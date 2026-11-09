"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

COMBINED STRATEGY:
  1. Fixed graded-index layer pattern (alternating L/H starting with L) - inspired by
     the high-scoring solution that uses a fixed pattern with multi-start Nelder-Mead.
  2. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  3. Multi-start Nelder-Mead optimization with diverse initial points.
  4. L-BFGS-B gradient-based refinement for final polish.
  5. Analytical quarter-wave initialization at band center.
  6. Explicit deadline guard: return the best feasible result before the limit.
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
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
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


def _constrain_thicknesses(ths, max_total, max_layer):
    """Constrain thicknesses to be within [0, max_layer] and sum <= max_total."""
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
    return ths


def _nelder_mead(f, x0, lo, hi, deadline, max_iters=300):
    """Nelder-Mead local optimization with box constraints."""
    n = len(x0)
    best = list(x0)
    best_val = f(best)
    
    simplex = [list(x0)]
    step = 0.1 * (hi - lo)
    for i in range(n):
        pt = list(x0)
        pt[i] = min(hi[i], pt[i] + step[i])
        simplex.append(pt)
    
    scores = [f(pt) for pt in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    it = 0
    while it < max_iters and time.time() < deadline:
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


def _make_evaluator(materials, problem, max_total, max_layer):
    """Return (eval, lo, hi) for independent log-scale thicknesses."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    n = len(materials)
    lo = np.full(n, math.log(1.0))
    hi = np.full(n, math.log(max_layer))

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        return _mean_reflectance(materials, list(t), problem)

    return eval_x, lo, hi


def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before the 120 s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]

    best_ths = None
    best_mats = None
    best_score = float("inf")

    def maybe_record(mats, ths):
        nonlocal best_ths, best_mats, best_score
        ths = _constrain_thicknesses(ths, max_total, max_layer)
        if any(t < 0 or t > max_layer for t in ths):
            return
        if len(ths) > max_layers:
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)

    # --- Step 1: Define fixed graded-index layer pattern ---
    # Alternating L/H pattern starting with L (closer to air)
    n_layers = min(20, max_layers)
    materials = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append('L')
        else:
            materials.append('H')
    
    # --- Step 2: Initialize with analytical quarter-wave solution ---
    n_mat = {k: v for k, v in problem["materials"].items()}
    init_ths = []
    for m in materials:
        n = n_mat[m]
        qw = wl_center / (4.0 * n)
        init_ths.append(qw)
    init_ths = _constrain_thicknesses(init_ths, max_total, max_layer)
    maybe_record(materials, init_ths)
    
    # --- Step 3: Multi-start Nelder-Mead on physical thicknesses ---
    eval_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)
    
    # Multiple diverse starting points
    n_starts = 15
    for start_i in range(n_starts):
        if time.time() >= deadline - 15:
            break
        
        # Generate perturbed starting point
        x0 = []
        for i, m in enumerate(materials):
            base_t = init_ths[i]
            # Random perturbation: scale factor between 0.5 and 2.0
            factor = random.uniform(0.5, 2.0)
            t = base_t * factor
            t = max(1.0, min(max_layer, t))
            x0.append(math.log(t))
        x0 = np.array(x0)
        
        # Run Nelder-Mead
        xc, val = _nelder_mead(eval_x, x0, lo, hi, deadline - 15, max_iters=400)
        
        # Convert back to physical thicknesses
        t = np.exp(np.clip(xc, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        maybe_record(materials, list(t))
    
    # --- Step 4: L-BFGS-B for gradient-based refinement ---
    if _HAVE_SCIPY and time.time() < deadline - 10:
        if best_ths is not None:
            x0 = [math.log(max(1.0, t)) for t in best_ths]
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 300, "maxfun": 5000, "maxls": 50,
                         "ftol": 1e-12, "gtol": 1e-10},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))
    
    # --- Step 5: Try alternative pattern (H/L starting with H) ---
    if time.time() < deadline - 10:
        materials_alt = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials_alt.append('H')
            else:
                materials_alt.append('L')
        
        init_ths_alt = []
        for m in materials_alt:
            n = n_mat[m]
            qw = wl_center / (4.0 * n)
            init_ths_alt.append(qw)
        init_ths_alt = _constrain_thicknesses(init_ths_alt, max_total, max_layer)
        maybe_record(materials_alt, init_ths_alt)
        
        eval_alt, lo_alt, hi_alt = _make_evaluator(materials_alt, problem, max_total, max_layer)
        
        n_starts_alt = 8
        for start_i in range(n_starts_alt):
            if time.time() >= deadline - 10:
                break
            
            x0 = []
            for i, m in enumerate(materials_alt):
                base_t = init_ths_alt[i]
                factor = random.uniform(0.5, 2.0)
                t = base_t * factor
                t = max(1.0, min(max_layer, t))
                x0.append(math.log(t))
            x0 = np.array(x0)
            
            xc, val = _nelder_mead(eval_alt, x0, lo_alt, hi_alt, deadline - 10, max_iters=300)
            
            t = np.exp(np.clip(xc, lo_alt, hi_alt))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials_alt, list(t))
    
    # --- Step 6: Final polish on the best solution ---
    if best_ths is not None and time.time() < deadline - 5:
        eval_final, lo_final, hi_final = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0 = [math.log(max(1.0, t)) for t in best_ths]
        
        # Nelder-Mead polish
        xc, val = _nelder_mead(eval_final, x0, lo_final, hi_final, deadline - 3, max_iters=500)
        t = np.exp(np.clip(xc, lo_final, hi_final))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        maybe_record(best_mats, list(t))
        
        # L-BFGS-B polish if available
        if _HAVE_SCIPY and time.time() < deadline - 2:
            res = _scipy_minimize(
                eval_final, x0, method="L-BFGS-B", bounds=list(zip(lo_final, hi_final)),
                options={"maxiter": 300, "maxfun": 5000, "maxls": 50,
                         "ftol": 1e-12, "gtol": 1e-10},
            )
            xc = np.clip(res.x, lo_final, hi_final)
            t = np.exp(np.clip(xc, lo_final, hi_final))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        qw_L = wl_center / (4.0 * nL)
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
