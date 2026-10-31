"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of unit-area equilateral triangle

# Vertices of the triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _triangle_area(p, q, r):
    """Area of triangle formed by points p, q, r."""
    return 0.5 * abs(np.cross(q - p, r - p))


def _min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-10 or y > H + 1e-10:
        return False
    # Check against each edge using cross products
    # Edge 0->1: y >= 0
    if y < -1e-10:
        return False
    # Edge 1->2: line from (1,0) to (0.5, H)
    # Direction: (-0.5, H), normal pointing inward: (H, 0.5)
    # Point should be on the side where (x-1)*H + y*0.5 <= 0 ... let's use barycentric
    # Barycentric coordinates
    # A = V0, B = V1, C = V2
    # P = A + u*(B-A) + v*(C-A), with u>=0, v>=0, u+v<=1
    u = x
    v = y / H
    if u < -1e-10 or v < -1e-10 or u + v > 1 + 1e-10:
        return False
    return True


def _objective(params):
    """Negative minimum triangle area (to minimize)."""
    points = params.reshape(N, 2)
    return -_min_triangle_area(points)


def _soft_min_objective(params, alpha=50.0):
    """
    Smooth surrogate: soft-min of all triangle areas.
    Uses log-sum-exp trick for numerical stability.
    We want to maximize the min area, so minimize -softmin.
    softmin(a_1,...,a_m) ≈ -log(sum(exp(-alpha*a_i)))/alpha
    So we minimize -softmin = log(sum(exp(-alpha*a_i)))/alpha
    """
    points = params.reshape(N, 2)
    n = len(points)
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs(np.cross(points[j] - points[i], points[k] - points[i]))
                areas.append(a)
    areas = np.array(areas)
    # Soft min via log-sum-exp
    neg_alpha_areas = -alpha * areas
    max_val = np.max(neg_alpha_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_alpha_areas - max_val)))
    soft_min = -log_sum / alpha
    return -soft_min  # negative because we're minimizing


def _penalized_objective(params, alpha=50.0, penalty_weight=1000.0):
    """
    Combined objective: -softmin(areas) + penalty for points outside triangle.
    """
    points = params.reshape(N, 2)
    
    # Compute all triangle areas
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs(np.cross(points[j] - points[i], points[k] - points[i]))
                areas.append(a)
    areas = np.array(areas)
    
    # Soft min
    neg_alpha_areas = -alpha * areas
    max_val = np.max(neg_alpha_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_alpha_areas - max_val)))
    soft_min = -log_sum / alpha
    
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        x, y = p
        u = x
        v = y / H
        if u < 0:
            penalty += u * u
        if v < 0:
            penalty += v * v
        if u + v > 1:
            penalty += (u + v - 1) * (u + v - 1)
        # Also penalize if y > H (shouldn't happen if u+v<=1 and u,v>=0 but just in case)
        if y > H:
            penalty += (y - H) * (y - H)
    
    return -soft_min + penalty_weight * penalty


def _project_to_triangle(points):
    """Project points to be inside the triangle using barycentric clipping."""
    result = np.zeros_like(points)
    for idx, p in enumerate(points):
        x, y = p
        u = x
        v = y / H
        # Clip to barycentric coordinates
        if u < 0:
            u = 0
        if v < 0:
            v = 0
        if u + v > 1:
            s = u + v
            u /= s
            v /= s
        result[idx] = [u + 0.5 * v, H * v]
    return result


def _initial_guess(seed):
    """Generate an initial guess with some structure."""
    rng = np.random.RandomState(seed)
    # Start with a grid-like arrangement plus jitter
    points = []
    # Use a simple lattice in barycentric coordinates
    # For 11 points, try a 3x3 grid with center removed or similar
    # Let's try: 4 points near corners/edges, 7 in interior
    # Simpler: random but with minimum distance constraint
    attempts = 0
    while len(points) < N and attempts < 10000:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            p = np.array([u + 0.5 * v, H * v])
            # Check minimum distance to existing points
            min_dist = float('inf')
            for q in points:
                d = np.linalg.norm(p - q)
                if d < min_dist:
                    min_dist = d
            if min_dist > 0.08 or len(points) < 3:  # allow closer for first few
                points.append(p)
        attempts += 1
    # If we didn't get enough, fill randomly
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        while u + v >= 1:
            u = rng.random()
            v = rng.random()
        points.append(np.array([u + 0.5 * v, H * v]))
    return np.array(points)


def construct():
    """
    Place 11 points in the unit-area equilateral triangle to maximize
    the minimum area of any triangle formed by three of the points.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple restarts with different initial guesses and optimization strategies
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + restart * 137
        points = _initial_guess(seed)
        params = points.flatten()
        
        # First, do a coarse optimization with soft-min surrogate
        alpha = 30.0
        result = minimize(
            _penalized_objective,
            params,
            args=(alpha, 1000.0),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            break
        
        # Project back to triangle
        points_opt = result.x.reshape(N, 2)
        points_opt = _project_to_triangle(points_opt)
        
        # Evaluate true objective
        score = _min_triangle_area(points_opt)
        if score > best_score:
            best_score = score
            best_points = points_opt.copy()
        
        # Second, refine with higher alpha (sharper soft-min)
        alpha2 = 100.0
        params2 = points_opt.flatten()
        result2 = minimize(
            _penalized_objective,
            params2,
            args=(alpha2, 10000.0),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if time.time() > deadline:
            break
        
        points_opt2 = result2.x.reshape(N, 2)
        points_opt2 = _project_to_triangle(points_opt2)
        
        score2 = _min_triangle_area(points_opt2)
        if score2 > best_score:
            best_score = score2
            best_points = points_opt2.copy()
        
        # Third, try direct minimization of negative min area (nonsmooth)
        # Use Nelder-Mead or similar for the nonsmooth objective
        params3 = points_opt2.flatten()
        result3 = minimize(
            _objective,
            params3,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if time.time() > deadline:
            break
        
        points_opt3 = result3.x.reshape(N, 2)
        points_opt3 = _project_to_triangle(points_opt3)
        
        score3 = _min_triangle_area(points_opt3)
        if score3 > best_score:
            best_score = score3
            best_points = points_opt3.copy()
    
    # Final safety: ensure we have valid points
    if best_points is None:
        # Fallback: return a reasonable configuration
        best_points = _initial_guess(42)
    
    # Final projection to ensure validity
    best_points = _project_to_triangle(best_points)
    
    return best_points
