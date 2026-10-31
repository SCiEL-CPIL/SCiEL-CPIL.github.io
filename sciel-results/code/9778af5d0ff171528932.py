"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

# Precompute all C(13,3)=286 triples
_TRIPLES = np.array(list(combinations(range(N), 3)), dtype=np.int64)

def _triangle_crosses(points):
    """Return cross products (twice the signed area) for all triples.
    points: (N, 2) array. Returns (286,) array of 2*area (signed)."""
    i = _TRIPLES[:, 0]
    j = _TRIPLES[:, 1]
    k = _TRIPLES[:, 2]
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    crosses = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return crosses

def _min_triangle_area(points):
    crosses = _triangle_crosses(points)
    return 0.5 * np.min(np.abs(crosses))

def _convex_hull_area(points):
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: shoelace on convex hull via sorting by angle
        # For degenerate cases, return a small value
        return 1e-10

def _score(points):
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _soft_min_abs(crosses, beta):
    """Smooth approximation of min(|crosses|) using log-sum-exp.
    Returns a smooth lower-bound-like surrogate."""
    abs_c = np.abs(crosses)
    # soft-min: -log-sum-exp(-beta * x) / beta approximates min(x) for large beta
    m = np.min(abs_c)
    shifted = abs_c - m
    s = np.sum(np.exp(-beta * shifted))
    return m - np.log(s) / beta

def _objective(params, beta):
    """Objective to minimize: -soft_min_triangle_area / convex_hull_area.
    params: (26,) flat array of 13 points."""
    points = params.reshape(N, 2)
    crosses = _triangle_crosses(points)
    soft_min = _soft_min_abs(crosses, beta)
    # Convex hull area via shoelace on sorted hull
    # For speed, approximate hull area using all points (overestimate if non-convex)
    # Better: compute convex hull area
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        hull_area = 1e-10
    if hull_area < 1e-15:
        return 1e10
    return -soft_min / hull_area

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    angles = angle_offset + 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _optimize_from_start(start_points, beta, max_iters, deadline):
    """Run Nelder-Mead / Powell optimization from a starting point."""
    from scipy.optimize import minimize
    
    params = start_points.flatten().copy()
    
    best_params = params.copy()
    best_score = _score(start_points)
    
    # Use Powell for derivative-free optimization
    result = minimize(
        lambda p: _objective(p, beta),
        params,
        method='Powell',
        options={'maxiter': max_iters, 'xtol': 1e-8, 'ftol': 1e-10},
    )
    
    if result.success or hasattr(result, 'x'):
        candidate = result.x.reshape(N, 2)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_params = result.x.copy()
    
    # Check deadline
    if time.time() > deadline - 5:
        return best_params, best_score
    
    # Polish with tighter tolerance
    result2 = minimize(
        lambda p: _objective(p, beta * 2),
        best_params,
        method='Nelder-Mead',
        options={'maxiter': max_iters // 2, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
    )
    if hasattr(result2, 'x'):
        candidate2 = result2.x.reshape(N, 2)
        cand_score2 = _score(candidate2)
        if cand_score2 > best_score:
            best_score = cand_score2
            best_params = result2.x.copy()
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    best_params = None
    best_score = -1.0
    
    # Generate diverse starting configurations
    rng = np.random.RandomState(42)
    
    starts = []
    # Regular polygon
    starts.append(_regular_polygon(N))
    # Regular polygon with different rotation
    starts.append(_regular_polygon(N, angle_offset=0.1))
    # Regular polygon with different radius
    starts.append(_regular_polygon(N, radius=0.8))
    # Jittered regular polygon
    for jitter_scale in [0.05, 0.1, 0.15, 0.2]:
        base = _regular_polygon(N)
        jittered = base + rng.randn(N, 2) * jitter_scale
        starts.append(jittered)
    # Random points in unit square (normalized)
    for _ in range(5):
        rand_pts = rng.rand(N, 2)
        # Center and scale to unit area-ish
        rand_pts -= rand_pts.mean(axis=0)
        scale = np.sqrt(np.mean(np.sum(rand_pts**2, axis=1)))
        if scale > 1e-10:
            rand_pts /= scale
        starts.append(rand_pts)
    # Fibonacci spiral on unit circle
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = np.array([[np.cos(i * golden_angle), np.sin(i * golden_angle)] for i in range(N)])
    starts.append(fib_pts)
    
    # Try each start
    beta = 20.0  # soft-min parameter
    max_iters = 200
    
    for idx, start in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        params, score = _optimize_from_start(start, beta, max_iters, deadline)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
        
        # If we're running low on time, break
        if time.time() > deadline - 15:
            break
    
    # Final polish if we have time
    if best_params is not None and time.time() < deadline - 5:
        from scipy.optimize import minimize
        # Try a few more local refinements
        for polish_beta in [50.0, 100.0]:
            if time.time() > deadline - 3:
                break
            result = minimize(
                lambda p: _objective(p, polish_beta),
                best_params,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12},
            )
            if hasattr(result, 'x'):
                cand = result.x.reshape(N, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_params = result.x.copy()
    
    if best_params is None:
        # Fallback: regular polygon
        best_params = _regular_polygon(N).flatten()
    
    return best_params.reshape(N, 2)
# EVOLVE-BLOCK-END
