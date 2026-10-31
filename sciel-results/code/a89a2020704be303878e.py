"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 13

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    # Vectorized computation of all triangle areas
    # For each triple (i, j, k), area = 0.5 * |cross(pj - pi, pk - pi)|
    # We'll do it in chunks to avoid memory issues
    idx = np.arange(n)
    for i in range(n - 2):
        pi = points[i]
        # All pairs (j, k) with j > i, k > j
        rest = points[i+1:]
        m = len(rest)
        if m < 2:
            continue
        # Compute cross products for all pairs in rest
        # rest[j] - pi, rest[k] - pi
        # cross = (rest[j][0]-pi[0])*(rest[k][1]-pi[1]) - (rest[j][1]-pi[1])*(rest[k][0]-pi[0])
        dx = rest[:, 0] - pi[0]
        dy = rest[:, 1] - pi[1]
        # For each pair (a, b) in rest with a < b:
        # cross = dx[a]*dy[b] - dy[a]*dx[b]
        # Use broadcasting
        cross = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
        # Only take upper triangle (a < b)
        triu_mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        areas = 0.5 * np.abs(cross[triu_mask])
        local_min = areas.min()
        if local_min < min_area:
            min_area = local_min
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_points = None
    best_score = -float('inf')
    
    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s
    
    # Initial guess: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(initial)
    
    # Strategy 1: Optimize radial and angular perturbations around regular polygon
    # Parameters: 13 radial values (multipliers on radius) + 13 angular shifts
    def params_to_points(params):
        r = 1.0 + params[:N] * 0.5  # Radial perturbation, scaled to keep reasonable
        theta = angles + params[N:] * 0.5  # Angular perturbation
        # Ensure convexity by sorting by angle
        order = np.argsort(theta)
        theta_sorted = theta[order]
        r_sorted = r[order]
        # Renormalize angles to be in [0, 2*pi)
        theta_sorted = np.mod(theta_sorted, 2 * np.pi)
        # Sort again after mod
        order2 = np.argsort(theta_sorted)
        theta_sorted = theta_sorted[order2]
        r_sorted = r_sorted[order2]
        pts = np.column_stack([r_sorted * np.cos(theta_sorted), r_sorted * np.sin(theta_sorted)])
        return pts
    
    def neg_score_from_params(params):
        if time.time() > deadline:
            return 1e10
        pts = params_to_points(params)
        return _negative_score(pts)
    
    # Differential evolution for the parameter space
    bounds = [(-0.3, 0.3)] * (2 * N)
    
    try:
        result = differential_evolution(
            neg_score_from_params,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=True,
            init='sobol',
            workers=1
        )
        if time.time() < deadline:
            if result.success or result.fun < 1e9:
                pts = params_to_points(result.x)
                update_best(pts)
    except Exception:
        pass
    
    # Strategy 2: Direct optimization of all 26 coordinates with BFGS
    if time.time() < deadline - 30:
        def neg_score_direct(pts_flat):
            if time.time() > deadline:
                return 1e10
            pts = pts_flat.reshape(N, 2)
            return _negative_score(pts)
        
        x0 = initial.flatten()
        try:
            result2 = minimize(
                neg_score_direct,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if time.time() < deadline:
                pts = result2.x.reshape(N, 2)
                update_best(pts)
        except Exception:
            pass
    
    # Strategy 3: Local search with random restarts
    if time.time() < deadline - 20:
        rng = np.random.RandomState(42)
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Start from regular polygon with small random perturbation
            angles_r = angles + rng.uniform(-0.1, 0.1, N)
            r_r = 1.0 + rng.uniform(-0.1, 0.1, N)
            order = np.argsort(angles_r)
            angles_r = angles_r[order]
            r_r = r_r[order]
            x0 = np.column_stack([r_r * np.cos(angles_r), r_r * np.sin(angles_r)]).flatten()
            
            try:
                result3 = minimize(
                    neg_score_direct,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if time.time() < deadline:
                    pts = result3.x.reshape(N, 2)
                    update_best(pts)
            except Exception:
                pass
    
    # Final: try a few more random starts with direct optimization
    if time.time() < deadline - 15:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Random points in unit circle
            r = np.sqrt(rng.uniform(0, 1, N))
            theta = rng.uniform(0, 2 * np.pi, N)
            order = np.argsort(theta)
            theta = theta[order]
            r = r[order]
            x0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).flatten()
            
            try:
                result4 = minimize(
                    neg_score_direct,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if time.time() < deadline:
                    pts = result4.x.reshape(N, 2)
                    update_best(pts)
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular 13-gon
        best_points = initial
    
    return best_points

# Keep the function name as required
construct = construct
