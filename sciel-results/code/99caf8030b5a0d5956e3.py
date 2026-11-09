"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRATEGY:
  1. Fast vectorized characteristic-matrix reflectance model.
  2. Differential Evolution (global) + L-BFGS-B (local polish) multi-start.
  3. Independent log-scale thickness parameterization for full flexibility.
  4. Explore a range of layer counts with diverse starting points.
  5. Final high-precision polish on the best solution found.
  6. Explicit deadline guard: return the best feasible result before the limit.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    from scipy.optimize import differential_evolution as _scipy_de
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


def _constrain_thicknesses(ths, max_total, max_layer):
    """Constrain thicknesses to be within [0, max_layer] and sum <= max_total."""
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
    return ths


def _make_evaluator(materials, problem, max_total, max_layer):
    """Return (eval, lo, hi) for independent log-scale thicknesses."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
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
    deadline = start_time + 140  # safety margin before the 150 s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    qw_L = wl_center / (4.0 * nL)
    qw_H = wl_center / (4.0 * nH)

    best_ths = None
    best_mats = None
    best_score = float("inf")

    def maybe_record(mats, ths):
        nonlocal best_ths, best_mats, best_score
        ths = _constrain_thicknesses(ths, max_total, max_layer)
        if any(t < 0 or t > max_layer for t in ths):
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)

    def _optimize_independent(n_layers, n_starts=8):
        """Optimize independent thicknesses for a given layer count."""
        nonlocal best_ths, best_mats, best_score
        if time.time() >= deadline - 15:
            return
        materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        eval_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)

        # Try Differential Evolution first for global exploration
        if _HAVE_SCIPY and time.time() < deadline - 30:
            try:
                de_res = _scipy_de(
                    eval_x, bounds=list(zip(lo, hi)),
                    maxiter=50, tol=1e-6, seed=42,
                    polish=False, popsize=15, mutation=(0.5, 1.0),
                    recombination=0.8, init='latinhypercube'
                )
                xc = np.clip(de_res.x, lo, hi)
                t = np.exp(np.clip(xc, lo, hi))
                total = t.sum()
                if total > max_total:
                    t = t * (max_total / total)
                maybe_record(materials, list(t))
            except Exception:
                pass

        # Multiple diverse starting points with L-BFGS-B
        for start_i in range(n_starts):
            if time.time() >= deadline - 10:
                return

            # Generate starting point
            x0 = []
            for m in materials:
                qw = qw_L if m == 'L' else qw_H
                factor = random.uniform(0.3, 2.0)
                t = qw * factor
                t = max(1.0, min(max_layer, t))
                x0.append(math.log(t))
            x0 = np.array(x0)

            if _HAVE_SCIPY:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 300, "maxfun": 5000, "maxls": 40},
                )
                xc = np.clip(res.x, lo, hi)
            else:
                # Simple gradient-free fallback
                xc = x0.copy()
                for _ in range(200):
                    if time.time() >= deadline - 5:
                        break
                    step = 0.05 * random.uniform(0.5, 1.5)
                    x_new = xc + step * np.random.randn(len(xc))
                    x_new = np.clip(x_new, lo, hi)
                    if eval_x(list(x_new)) < eval_x(list(xc)):
                        xc = x_new
                xc = np.clip(xc, lo, hi)

            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))

    # Explore layer counts
    counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    counts = [c for c in counts if c <= max_layers]

    # Interleave small and large counts for early fallback
    order = []
    i, j = 0, len(counts) - 1
    while i <= j:
        order.append(counts[i])
        if i != j:
            order.append(counts[j])
        i += 1
        j -= 1

    for n_layers in order:
        if time.time() >= deadline - 20:
            break
        n_starts = 10 if n_layers <= 8 else 6
        _optimize_independent(n_layers, n_starts)

    # Final high-precision polish on the best solution found
    if best_ths is not None and time.time() < deadline - 10:
        eval_x, lo, hi = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0 = [math.log(max(1.0, t)) for t in best_ths]

        if _HAVE_SCIPY:
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 500, "maxfun": 10000, "maxls": 50,
                         "ftol": 1e-14, "gtol": 1e-12},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
