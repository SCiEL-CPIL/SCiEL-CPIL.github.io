# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(pts):
    """Return (d_min/d_max)^2, or 0 on degenerate input."""
    try:
        d = pdist(pts)
        if d.size == 0:
            return 0.0
        dmin = d.min()
        dmax = d.max()
        if dmax <= 1e-12 or not np.isfinite(dmin) or not np.isfinite(dmax):
            return 0.0
        r = dmin / dmax
        return float(r * r)
    except Exception:
        return 0.0

def _neg_score(pts_flat):
    pts = pts_flat.reshape(14, 3)
    return -_score(pts)

def _icosahedron_vertices():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0, 1,  phi], [0, -1, -phi], [0, 1, -phi],
        [ phi, 0, -1], [ phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n

def _fibonacci_sphere(n):
    pts = np.empty((n, 3))
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = 2.0 * np.pi * i / golden
        pts[i] = [np.cos(theta) * r, y, np.sin(theta) * r]
    return pts

def _initial_configs():
    """Yield a list of candidate 14x3 starting configurations."""
    configs = []
    ico = _icosahedron_vertices()  # 12 points on unit sphere
    # Config A: icosahedron + center + one extra on sphere
    for extra_theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        extra = np.array([np.cos(extra_theta), 0.0, np.sin(extra_theta)])
        p = np.vstack([ico, [[0.0, 0.0, 0.0]], [extra]])
        configs.append(p.copy())
    # Config B: icosahedron scaled to 0.8 + center + one at radius 1.2
    ico_s = ico * 0.8
    for extra_theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        extra = np.array([1.2 * np.cos(extra_theta), 0.0, 1.2 * np.sin(extra_theta)])
        p = np.vstack([ico_s, [[0.0, 0.0, 0.0]], [extra]])
        configs.append(p.copy())
    # Config C: fibonacci sphere 14 points
    configs.append(_fibonacci_sphere(14).copy())
    # Config D: icosahedron + 2 points at opposite poles
    p = np.vstack([ico, [[0.0, 0.0, 1.3]], [[0.0, 0.0, -1.3]]])
    configs.append(p.copy())
    return configs

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 560  # safety margin before 630s kill

    best_pts = None
    best_val = -1.0

    configs = _initial_configs()

    # Try each initial config with Nelder-Mead
    for idx, cfg in enumerate(configs):
        if time.time() > deadline:
            break
        x0 = cfg.flatten()
        try:
            res = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'maxfev': 6000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_pts = res.x.reshape(14, 3).copy()
        except Exception:
            pass

        # Also evaluate the raw config in case optimization didn't improve
        raw_val = _score(cfg)
        if raw_val > best_val:
            best_val = raw_val
            best_pts = cfg.copy()

    # If we still have time, do a second round of refinement from the best
    if best_pts is not None and time.time() < deadline - 30:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_pts = res.x.reshape(14, 3).copy()
        except Exception:
            pass

    # Final safety: if best_pts is None or invalid, fall back to a known-good config
    if best_pts is None:
        best_pts = configs[0].copy()

    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = configs[0].copy()

    return best_pts

# EVOLVE-BLOCK-END
