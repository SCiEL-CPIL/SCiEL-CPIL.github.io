"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _all_triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    n = len(pts)
    # Get all combinations
    idx = list(itertools.combinations(range(n), 3))
    idx = np.array(idx)
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    ab = b - a
    ac = c - a
    areas = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    return areas

def _hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    hull = _hull_area(pts)
    if hull < 1e-10:
        return 0.0
    return min_area / hull

def _soft_obj(pts_flat, t):
    """Differentiable soft-min objective for L-BFGS-B."""
    pts = pts_flat.reshape(N, 2)
    areas = _all_triangle_areas(pts)
    hull = _hull_area(pts)
    if hull < 1e-10:
        return 1e10
    neg_areas = -areas / t
    max_neg = np.max(neg_areas)
    logsumexp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -logsumexp * t
    return -soft_min / hull

def _neg_score(pts_flat):
    """Hard objective: negative score for final polish."""
    pts = pts_flat.reshape(N, 2)
    s = _score(pts)
    return -s

def _initial_regular_polygon(n, radius=1.0):
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _initial_hexagonal(n, seed=0):
    """Points on a hexagonal grid pattern."""
    rng = np.random.RandomState(seed)
    # Start with a hexagonal lattice, then jitter
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for row in range(side):
        for col in range(side):
            if len(pts) >= n:
                break
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts[:n])
    # Normalize to unit area
    hull = _hull_area(pts)
    if hull > 1e-10:
        pts = pts / np.sqrt(hull)
    # Add small random perturbation
    pts = pts + rng.normal(0, 0.01, pts.shape)
    return pts

def _initial_cosine_mod(n, k=2, amp=0.15, seed=0):
    """Regular polygon with radial modulation."""
    angles = 2 * np.pi * np.arange(n) / n
    radii = 1.0 + amp * np.cos(k * angles + seed * 0.1)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _initial_mixed(n, seed=0):
    """Mix of ring and interior points."""
    rng = np.random.RandomState(seed)
    n_ring = n // 2
    n_inner = n - n_ring
    # Ring points
    ring_angles = 2 * np.pi * np.arange(n_ring) / n_ring + rng.uniform(0, 0.3)
    ring_pts = np.column_stack([np.cos(ring_angles), np.sin(ring_angles)])
    # Inner points
    inner_r = rng.uniform(0.1, 0.5, n_inner)
    inner_theta = rng.uniform(0, 2 * np.pi, n_inner)
    inner_pts = np.column_stack([inner_r * np.cos(inner_theta), inner_r * np.sin(inner_theta)])
    pts = np.vstack([ring_pts, inner_pts])
    return pts

def _normalize_unit_area(pts):
    """Scale points so convex hull has unit area."""
    hull = _hull_area(pts)
    if hull > 1e-10:
        pts = pts / np.sqrt(hull)
    return pts

def _lbfgs_optimize(pts, t, maxiter=300, deadline=None):
    """Run L-BFGS-B with soft-min objective."""
    def obj(x):
        return _soft_obj(x, t)
    try:
        result = minimize(
            obj, pts.flatten(), method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x.reshape(N, 2)
    except Exception:
        return pts

def _nm_polish(pts, deadline=None):
    """Nelder-Mead polish on hard score."""
    try:
        result = minimize(
            _neg_score, pts.flatten(), method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        return result.x.reshape(N, 2)
    except Exception:
        return pts

def _coordinate_descent_polish(pts, deadline=None):
    """Simple coordinate descent: optimize each point's x and y separately."""
    best = pts.copy()
    best_s = _score(best)
    n_iters = 30
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small perturbations
                for delta in [0.001, -0.001, 0.01, -0.01, 0.005, -0.005]:
                    trial = best.copy()
                    trial[i, coord] += delta
                    s = _score(trial)
                    if s > best_s:
                        best = trial
                        best_s = s
                        improved = True
                        break
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_pts = None
    
    # Generate diverse initial configurations
    configs = []
    configs.append(_initial_regular_polygon(N, 1.0))
    configs.append(_initial_regular_polygon(N, 1.5))
    configs.append(_initial_hexagonal(N, seed=0))
    configs.append(_initial_hexagonal(N, seed=42))
    configs.append(_initial_cosine_mod(N, k=2, amp=0.15, seed=0))
    configs.append(_initial_cosine_mod(N, k=3, amp=0.1, seed=1))
    configs.append(_initial_cosine_mod(N, k=4, amp=0.2, seed=2))
    configs.append(_initial_mixed(N, seed=0))
    configs.append(_initial_mixed(N, seed=100))
    
    # Normalize all to unit hull area
    configs = [_normalize_unit_area(c) for c in configs]
    
    # Phase 1: Coarse optimization with multiple temperatures
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        init_score = _score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_pts = init_pts.copy()
        
        # Multi-stage soft-min optimization
        for t in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 10:
                break
            opt_pts = _lbfgs_optimize(init_pts, t, maxiter=200, deadline=deadline)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
            init_pts = opt_pts
    
    # Phase 2: Polish the best result
    if best_pts is not None and time.time() < deadline - 15:
        # Nelder-Mead on hard score
        polished = _nm_polish(best_pts, deadline)
        p_score = _score(polished)
        if p_score > best_score:
            best_score = p_score
            best_pts = polished.copy()
    
    # Phase 3: Coordinate descent polish
    if best_pts is not None and time.time() < deadline - 5:
        cd_pts = _coordinate_descent_polish(best_pts, deadline)
        cd_score = _score(cd_pts)
        if cd_score > best_score:
            best_score = cd_score
            best_pts = cd_pts.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = _initial_regular_polygon(N, 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
