"""Optimize 8 planar points maximizing product of pairwise distances at fixed diameter.
Uses central symmetry (4 points + negations) to halve search dimension,
with differential evolution + L-BFGS-B refinement and a deadline guard."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130  # 20s safety margin under 150s limit

def _remaining():
    return _DEADLINE - time.time()

def _build_points(p):
    """p is 8-dim vector: 4 points (x,y) each. Returns 8x2 array with central symmetry."""
    pts = np.array(p).reshape(4, 2)
    return np.vstack([pts, -pts])

def _score(pts):
    """Compute Delta/n^n after scaling to diameter 2. Returns log for stability."""
    n = len(pts)
    # All pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    # Product over ordered pairs i != j of distances
    # log(Delta) = sum_{i != j} log(d_ij)
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(dists[mask] + 1e-300))
    # Scale to diameter 2: divide all distances by current diameter
    # After scaling, each distance d -> d/D where D is diameter
    # There are n*(n-1) ordered pairs, so Delta_scaled = Delta / D^(n*(n-1))
    D = np.max(dists)
    if D < 1e-300:
        return -np.inf
    log_delta_scaled = log_delta - n * (n - 1) * np.log(D)
    # Score = Delta_scaled / n^n
    log_score = log_delta_scaled - n * np.log(n)
    return log_score

def _objective(p):
    """Negative log-score (minimize)."""
    pts = _build_points(p)
    # Check distinctness
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-12:
        return 1e10  # Penalty for coincident points
    s = _score(pts)
    if np.isneginf(s) or np.isnan(s):
        return 1e10
    return -s

def _objective_lbfgs(p):
    """Same as _objective but for L-BFGS-B (may be called with bounds)."""
    return _objective(p)

def construct():
    n = 8
    # Initial guess: regular octagon in centrally symmetric form
    # Regular octagon vertices at angles k*pi/4, k=0..7
    # Central symmetry: points at angles 0, pi/4, pi/2, 3pi/4 and their negatives
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    init = []
    for a in angles:
        init.extend([np.cos(a), np.sin(a)])
    init = np.array(init)

    best_log_score = _score(_build_points(init))  # Should be 0 (score=1)
    best_p = init.copy()

    # Differential evolution for global search
    # Bounds: coordinates in [-2, 2] (diameter will be scaled anyway, but keep reasonable)
    bounds = [(-2.0, 2.0)] * 8
    maxiter = 100
    popsize = 20

    if _remaining() > 30:
        try:
            result = differential_evolution(
                _objective, bounds,
                maxiter=maxiter, popsize=popsize,
                seed=42, tol=1e-10, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='latinhypercube',
                x0=init,
                callback=None
            )
            if _remaining() > 5:
                candidate = result.x
                s = _score(_build_points(candidate))
                if np.isfinite(s) and s > best_log_score:
                    best_log_score = s
                    best_p = candidate.copy()
        except Exception:
            pass

    # Local refinement with L-BFGS-B
    if _remaining() > 10:
        try:
            res = minimize(
                _objective_lbfgs, best_p,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if _remaining() > 3:
                candidate = res.x
                s = _score(_build_points(candidate))
                if np.isfinite(s) and s > best_log_score:
                    best_log_score = s
                    best_p = candidate.copy()
        except Exception:
            pass

    # Second round of DE from best point if time remains
    if _remaining() > 40:
        try:
            # Perturb best solution and run another DE
            np.random.seed(123)
            perturbed = best_p + np.random.normal(0, 0.1, size=8)
            perturbed = np.clip(perturbed, -2.0, 2.0)
            result2 = differential_evolution(
                _objective, bounds,
                maxiter=50, popsize=15,
                seed=123, tol=1e-10, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='latinhypercube',
                x0=perturbed
            )
            if _remaining() > 5:
                candidate = result2.x
                s = _score(_build_points(candidate))
                if np.isfinite(s) and s > best_log_score:
                    best_log_score = s
                    best_p = candidate.copy()
        except Exception:
            pass

    # Final polish
    if _remaining() > 5:
        try:
            res = minimize(
                _objective_lbfgs, best_p,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            candidate = res.x
            s = _score(_build_points(candidate))
            if np.isfinite(s) and s > best_log_score:
                best_log_score = s
                best_p = candidate.copy()
        except Exception:
            pass

    # Convert to list of [x, y]
    pts = _build_points(best_p)
    # Scale to diameter 2
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    D = np.max(dists)
    if D > 1e-300:
        pts = pts / D * 2.0

    # Ensure distinctness and valid coordinates
    result = [[float(pts[i, 0]), float(pts[i, 1])] for i in range(8)]

    # Verify distinctness
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(result[i][0] - result[j][0]) < 1e-10 and abs(result[i][1] - result[j][1]) < 1e-10:
                # Nudge
                result[j][0] += 1e-8

    return result
