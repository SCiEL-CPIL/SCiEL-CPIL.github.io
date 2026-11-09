"""Fifty equal circles in a square – optimised packing."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # safety margin before 120s kill


def _score_from_centres(c):
    """Compute the radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _negative_score(c_flat):
    """Objective to minimise (negative radius score)."""
    c = c_flat.reshape(N, 2)
    # Clamp to box for feasibility during optimisation
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    r = _score_from_centres(c)
    return -r


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def construct():
    t0 = time.time()
    best_centres = _grid_init()
    best_score = _score_from_centres(best_centres)

    rng = np.random.default_rng(42)

    # Strategy 1: Local optimisation from multiple random starts
    # Use L-BFGS-B or SLSQP with clipping
    n_starts = 12
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 30:
            break
        if i == 0:
            x0 = best_centres.ravel()
        else:
            x0 = _random_init(rng).ravel()

        # Perturb slightly if starting from best
        if i > 0:
            x0 = x0 + rng.normal(0, 0.02, size=x0.shape)

        try:
            res = minimize(
                _negative_score, x0,
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < 0:
                c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
        except Exception:
            pass

    # Strategy 2: Differential evolution for global search
    if time.time() - t0 < DEADLINE - 20:
        try:
            bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
            # Use a subset of points for DE to keep it feasible in time
            # Actually DE on 100 variables is expensive; do a limited run
            de_result = differential_evolution(
                _negative_score, bounds,
                seed=42, maxiter=50, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=False,
                callback=None
            )
            if de_result.fun < 0:
                c = np.clip(de_result.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
        except Exception:
            pass

    # Strategy 3: Refine with SLSQP from best
    if time.time() - t0 < DEADLINE - 5:
        try:
            res = minimize(
                _negative_score, best_centres.ravel(),
                method='SLSQP',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s = _score_from_centres(c)
            if s > best_score:
                best_score = s
                best_centres = c
        except Exception:
            pass

    # Strategy 4: Final L-BFGS-B polish
    if time.time() - t0 < DEADLINE - 2:
        try:
            res = minimize(
                _negative_score, best_centres.ravel(),
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s = _score_from_centres(c)
            if s > best_score:
                best_score = s
                best_centres = c
        except Exception:
            pass

    # Ensure valid output
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres
