"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Container constraint
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    # Pairwise constraint
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.hypot(dx, dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    # Try several spacings and pick the one that fits N points best
    best_pts = None
    for spacing in np.linspace(0.15, 0.35, 100):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by distance from center, take first N
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            if best_pts is None:
                best_pts = pts
            else:
                # Prefer the one with higher score
                if _score(pts) > _score(best_pts):
                    best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    return np.random.uniform(-0.8, 0.8, (N, 2))


def _optimize_from(pts0):
    """Run gradient-free optimization to maximize the packing radius."""
    def neg_score(x):
        pts = x.reshape(N, 2)
        s = _score(pts)
        return -s if s > 0 else 1.0  # Penalize invalid configurations
    
    # Use Nelder-Mead for robustness
    res = minimize(neg_score, pts0.flatten(), method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    return res.x.reshape(N, 2)


def construct():
    start_time = time.time()
    
    # Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # Run optimization with time guard
    try:
        res = _optimize_from(pts0)
        res_score = _score(res)
        if res_score > best_score:
            best_pts = res
            best_score = res_score
    except Exception:
        pass
    
    # If we have time, try a second refinement pass
    if time.time() - start_time < 120.0 and best_score > 0:
        try:
            res2 = _optimize_from(best_pts)
            res2_score = _score(res2)
            if res2_score > best_score:
                best_pts = res2
                best_score = res2_score
        except Exception:
            pass
    
    # Final safety check: ensure validity
    if _score(best_pts) <= 0:
        # Return the initial hex lattice as fallback
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
