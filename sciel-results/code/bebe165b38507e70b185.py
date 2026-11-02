"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0  # seconds before deadline to stop

# Precompute all triangle index triplets
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_NUM_TRI = len(_TRI_INDICES)


def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (num_tri,)."""
    # pts: (n, 2), _TRI_INDICES: (num_tri, 3)
    i = _TRI_INDICES[:, 0]
    j = _TRI_INDICES[:, 1]
    k = _TRI_INDICES[:, 2]
    v1 = pts[j] - pts[i]  # (num_tri, 2)
    v2 = pts[k] - pts[i]  # (num_tri, 2)
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas


def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume == area
    except Exception:
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _all_triangle_areas(pts)
    return areas.min() / hull_area


def _neg_score(x):
    """Objective for minimization: negative of the score."""
    pts = x.reshape(N, 2)
    return -_score(pts)


def _fermat_spiral_init(n, center=(0.5, 0.5), radius=0.4):
    """Fermat spiral initialization: points placed on a golden-angle spiral."""
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~2.39996
    # Use sqrt spacing for uniform area density
    r = radius * np.sqrt(np.arange(n) / n)
    theta = golden_angle * np.arange(n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def _regular_polygon_init(n, center=(0.5, 0.5), radius=0.4):
    """Place n points: regular (n-1)-gon around center + 1 center point."""
    m = n - 1
    angles = 2 * np.pi * np.arange(m) / m
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    pts = np.vstack([pts, np.array(center)])
    return pts


def _simulated_annealing(pts, start_temp, end_temp, n_iterations, rng, time_limit_fn):
    """Simulated annealing targeting the worst triangle."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    for it in range(n_iterations):
        if time_limit_fn():
            break
        
        # Pick a random point to move
        idx = rng.randint(N)
        
        # Propose a move: random perturbation scaled by temperature
        scale = start_temp * (end_temp / start_temp) ** (it / max(1, n_iterations - 1))
        delta = rng.randn(2) * scale
        new_pts = current_pts.copy()
        new_pts[idx] += delta
        
        new_score = _score(new_pts)
        
        # Accept if better, or with probability exp((new - current) / temp)
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            temp = start_temp * (end_temp / start_temp) ** (it / max(1, n_iterations - 1))
            if temp > 0 and rng.random() < np.exp((new_score - current_score) / temp):
                current_pts = new_pts
                current_score = new_score
    
    return best_pts, best_score


def construct():
    global _START, _DEADLINE
    
    def _time_exceeded():
        return (time.time() - _START) > (_DEADLINE - _SAFETY)
    
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    if remaining < 5.0:
        return _regular_polygon_init(N)

    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    inits = [
        _fermat_spiral_init(N, radius=0.4),
        _fermat_spiral_init(N, radius=0.45),
        _fermat_spiral_init(N, radius=0.35),
        _regular_polygon_init(N, radius=0.4),
        _regular_polygon_init(N, radius=0.45),
        _regular_polygon_init(N, radius=0.35),
    ]
    
    # Add perturbed versions
    for base in [_fermat_spiral_init(N, radius=0.4), _regular_polygon_init(N, radius=0.4)]:
        for _ in range(2):
            inits.append(base + rng.randn(N, 2) * 0.03)
    
    # Evaluate all inits and find the best
    best_pts = inits[0]
    best_score = _score(best_pts)
    for init in inits[1:]:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    # Phase 1: Simulated annealing from the best init
    if not _time_exceeded():
        sa_pts, sa_score = _simulated_annealing(
            best_pts, start_temp=0.05, end_temp=0.001,
            n_iterations=500, rng=rng, time_limit_fn=_time_exceeded
        )
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
    
    # Phase 2: Nelder-Mead refinement from the best SA result
    if not _time_exceeded():
        x0 = best_pts.flatten()
        time_check = {'stop': False}
        
        def _callback(xk):
            if _time_exceeded():
                time_check['stop'] = True
            return time_check['stop']
        
        try:
            result = minimize(_neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
                              callback=_callback)
            if not time_check['stop']:
                candidate = result.x.reshape(N, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_pts = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: If time remains, try SA from a different good init + NM refinement
    if not _time_exceeded() and (time.time() - _START) < (_DEADLINE - _SAFETY - 20):
        # Pick a different good starting point
        alt_init = _fermat_spiral_init(N, radius=0.42)
        sa2_pts, sa2_score = _simulated_annealing(
            alt_init, start_temp=0.04, end_temp=0.001,
            n_iterations=300, rng=rng, time_limit_fn=_time_exceeded
        )
        if sa2_score > best_score:
            best_score = sa2_score
            best_pts = sa2_pts.copy()
        
        if not _time_exceeded():
            x0 = best_pts.flatten()
            time_check['stop'] = False
            
            def _callback2(xk):
                if _time_exceeded():
                    time_check['stop'] = True
                return time_check['stop']
            
            try:
                result2 = minimize(_neg_score, x0, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
                                   callback=_callback2)
                if not time_check['stop']:
                    candidate2 = result2.x.reshape(N, 2)
                    s2 = _score(candidate2)
                    if s2 > best_score:
                        best_score = s2
                        best_pts = candidate2.copy()
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
