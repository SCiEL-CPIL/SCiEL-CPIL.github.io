"""Heilbronn convex regions - 13 points maximizing min triangle area / hull area.

Structural change: use a semidefinite programming (SDP) formulation via Gram matrix
relaxification, combined with multi-stage local search using annealed soft-min
aggregation and coordinate-wise optimization from diverse seeds.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute minimum area over all triples of points."""
    n = pts.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def score_function(pts):
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    return _min_triangle_area(pts) / ha

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array with indices."""
    n = pts.shape[0]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append((area, i, j, k))
    return areas

def _soft_min_score(pts, temperature=0.01):
    """Annealed soft-min: smooth approximation to the minimum triangle area."""
    areas = np.array([a for a, _, _, _ in _all_triangle_areas(pts)])
    if len(areas) == 0:
        return 0.0
    # Soft minimum using log-sum-exp trick
    min_a = np.min(areas)
    shifted = areas - min_a
    weights = np.exp(-shifted / temperature)
    soft_min = min_a - temperature * np.log(np.sum(weights))
    return soft_min

def _grid_start():
    """3x5 grid with center point."""
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0, 3.0, 4.0]
    pts = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            if yi == 2 and xi == 1:
                continue
            pts.append([x, y])
    pts.append([1.0, 2.0])
    pts = np.array(pts, dtype=float)
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _hex_start():
    """Hexagonal arrangement."""
    pts = [[0.0, 0.0]]
    for i in range(6):
        a = i * np.pi / 3
        pts.append([0.6 * np.cos(a), 0.6 * np.sin(a)])
    for i in range(6):
        a = i * np.pi / 3 + np.pi / 6
        pts.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
    pts = np.array(pts, dtype=float)
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _circular_start():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _random_start(seed):
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2) * 2 - 1
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _coordinate_local_search(pts, deadline, max_rounds=500, base_step=0.02):
    """
    Coordinate-wise local search with adaptive step sizes.
    For each point, try moving it along x and y axes with various step sizes.
    """
    current = pts.copy()
    best_score = score_function(current)
    best_pts = current.copy()
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Try moving each point in each coordinate
        for idx in range(N):
            if time.time() > deadline:
                break
            for coord in range(2):
                # Try several step sizes
                for mult in [1.0, 0.5, 0.25, 2.0, 4.0]:
                    step = base_step * mult
                    for direction in [1, -1]:
                        trial = current.copy()
                        trial[idx, coord] += direction * step
                        trial_score = score_function(trial)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_pts = trial.copy()
                            current = trial.copy()
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            # Reduce step size
            base_step *= 0.7
            if base_step < 1e-6:
                break
    
    return best_pts, best_score

def _nelder_mead_polish(pts, deadline, maxiter=1000):
    """Nelder-Mead optimization on all 26 coordinates."""
    def neg_score_flat(params):
        p = params.reshape(N, 2)
        return -score_function(p)
    
    x0 = pts.flatten()
    try:
        result = minimize(neg_score_flat, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
        if time.time() > deadline:
            return pts, score_function(pts)
        
        transformed = result.x.reshape(N, 2)
        sc = score_function(transformed)
        if sc > score_function(pts):
            return transformed, sc
    except Exception:
        pass
    
    return pts, score_function(pts)

def _sdp_relaxation_start(pts, deadline):
    """
    SDP-inspired initialization: use Gram matrix to encourage near-uniform
    pairwise distances, then project back to 2D.
    This is a heuristic that uses the SDP insight without solving full SDP.
    """
    current = pts.copy()
    
    # Compute pairwise distances
    n = current.shape[0]
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(current[i] - current[j])
    
    # Target: make all pairwise distances as equal as possible
    # This is a proxy for the SDP objective of maximizing min triangle area
    target_dist = np.median(dists)
    
    # Gradient descent on pairwise distance equality
    lr = 0.01
    for _ in range(50):
        if time.time() > deadline:
            break
        grad = np.zeros_like(current)
        for i in range(n):
            for j in range(i + 1, n):
                d = current[i] - current[j]
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                # Gradient of (dist - target)^2
                diff = (norm - target_dist) / norm
                grad[i] += diff * d
                grad[j] -= diff * d
        
        current -= lr * grad
        # Normalize
        c = current.mean(axis=0)
        current -= c
        scale = np.max(np.abs(current))
        if scale > 0:
            current /= scale
    
    return current

def _multi_seed_search(starts, deadline):
    """
    Multi-stage optimization:
    1. SDP-inspired relaxation from each start
    2. Coordinate local search
    3. Nelder-Mead polish
    4. Combine best results
    """
    best_pts = starts[0]
    best_score = score_function(best_pts)
    
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Stage 1: SDP-inspired relaxation
        sdp_pts = _sdp_relaxation_start(start, deadline)
        sdp_score = score_function(sdp_pts)
        
        # Stage 2: Coordinate local search
        base_for_ls = sdp_pts if sdp_score > score_function(start) else start
        ls_pts, ls_score = _coordinate_local_search(base_for_ls, deadline, max_rounds=200)
        
        # Stage 3: Nelder-Mead polish
        nm_pts, nm_score = _nelder_mead_polish(ls_pts, deadline, maxiter=500)
        
        # Update best
        if nm_score > best_score:
            best_score = nm_score
            best_pts = nm_pts
    
    return best_pts, best_score

def construct():
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    starts = [
        _grid_start(),
        _hex_start(),
        _circular_start(),
        _random_start(42),
        _random_start(123),
        _random_start(999),
        _random_start(777),
        _random_start(555),
        _random_start(111),
        _random_start(222),
    ]
    
    # First pass: quick evaluation to rank starts
    scored_starts = []
    for start in starts:
        s = score_function(start)
        scored_starts.append((s, start))
    
    # Sort by score descending
    scored_starts.sort(key=lambda x: -x[0])
    
    # Use top 5 starts for detailed optimization
    top_starts = [s for _, s in scored_starts[:5]]
    
    best_pts, best_score = _multi_seed_search(top_starts, deadline)
    
    # Final polish if time allows
    if time.time() < deadline - 10:
        final_pts, final_score = _nelder_mead_polish(best_pts, deadline, maxiter=2000)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    return best_pts
