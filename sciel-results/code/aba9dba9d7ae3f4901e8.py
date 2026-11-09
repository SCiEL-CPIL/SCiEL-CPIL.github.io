import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50
LIMIT = 120.0  # seconds


def _radius(pts):
    """Compute the largest common radius for the given centres."""
    # Wall distance: min over i,k of (0.5 - |c_ik|)
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    # Avoid diagonal
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)


def _neg_radius(pts):
    return -_radius(pts)


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Grid start
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    best_pts = grid.copy()
    best_r = _radius(best_pts)

    # Nelder-Mead refinement from grid
    try:
        res = minimize(_neg_radius, best_pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        cand = res.x.reshape(N, 2)
        r = _radius(cand)
        if r > best_r:
            best_r = r
            best_pts = cand
    except Exception:
        pass

    if time.time() - t0 > LIMIT - 30:
        return best_pts

    # Multi-start: try several random initializations and refine
    n_starts = 5
    for s in range(n_starts):
        if time.time() - t0 > LIMIT - 30:
            break
        # Random initial points inside the square
        pts = rng.uniform(-0.5, 0.5, size=(N, 2))
        # Simple local push-apart to avoid coincident points
        for _ in range(50):
            d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
            np.fill_diagonal(d, np.inf)
            for i in range(N):
                for j in range(i + 1, N):
                    if d[i, j] < 0.02:
                        diff = pts[i] - pts[j]
                        if np.linalg.norm(diff) < 1e-8:
                            diff = rng.normal(size=2)
                        diff /= np.linalg.norm(diff)
                        pts[i] += 0.5 * diff * 0.02
                        pts[j] -= 0.5 * diff * 0.02
        # Clip to square
        pts = np.clip(pts, -0.49, 0.49)

        # Nelder-Mead from this start
        try:
            res = minimize(_neg_radius, pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            cand = res.x.reshape(N, 2)
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best_pts = cand
        except Exception:
            pass

    # Final polish: L-BFGS-B with bounds
    if time.time() - t0 < LIMIT - 15:
        try:
            bounds = [(-0.5, 0.5)] * (2 * N)
            res = minimize(_neg_radius, best_pts.flatten(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 5000, 'gtol': 1e-8})
            cand = res.x.reshape(N, 2)
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best_pts = cand
        except Exception:
            pass

    return best_pts
