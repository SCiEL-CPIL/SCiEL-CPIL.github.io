"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Pre-compute triangle indices for speed
def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRI = len(TRI_IDX)

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: (n, 2) array
    """
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def smooth_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns -softmin(areas) so that minimizing this maximizes the soft-min.
    """
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp trick for stability
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + beta * max_a
    softmin = -log_sum / beta + max_a
    return -softmin

def convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    """
    Calculate the score: min_triangle_area / convex_hull_area
    """
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def make_regular_polygon_plus_center(n):
    """Create a regular (n-1)-gon with center point."""
    if n <= 1:
        return np.array([[0.0, 0.0]])
    if n == 2:
        return np.array([[1.0, 0.0], [-1.0, 0.0]])
    
    m = n - 1  # number of polygon vertices
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    return np.vstack([poly, center])

def make_jittered_polygon(n, rng, jitter_scale=0.1):
    """Create a jittered regular polygon with center."""
    base = make_regular_polygon_plus_center(n)
    jitter = rng.random((n, 2)) * 2 - 1
    jittered = base + jitter * jitter_scale
    return jittered

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin of 5s before 150s kill
    
    best_score = -1.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular 12-gon + center
    initializations.append(make_regular_polygon_plus_center(N))
    
    # 2. Regular 13-gon (all on boundary)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 3. Jittered versions
    for js in [0.05, 0.1, 0.2, 0.3]:
        initializations.append(make_jittered_polygon(N, rng, js))
    
    # 4. Random points in unit circle
    for _ in range(3):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(pts)
    
    # 5. Grid-like pattern
    grid_x = np.array([-1, -0.5, 0, 0.5, 1])
    grid_y = np.array([-1, -0.5, 0, 0.5, 1])
    grid_pts = []
    for gx in grid_x:
        for gy in grid_y:
            if len(grid_pts) < N:
                grid_pts.append([gx, gy])
    if len(grid_pts) == N:
        initializations.append(np.array(grid_pts))
    
    def objective_smooth(x, beta=100.0):
        """Smooth objective for gradient-based optimization."""
        pts = x.reshape(-1, 2)
        return smooth_min_triangle_area(pts, beta)
    
    def objective_nonsmooth(x):
        """Nonsmooth objective for local polish."""
        pts = x.reshape(-1, 2)
        return -min_triangle_area(pts)
    
    def update_best(pts):
        nonlocal best_score, best_points
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Phase 1: Multi-scale smooth optimization
    # Start with coarse beta, then refine
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        for init in initializations:
            if time.time() > deadline - 10.0:
                break
            x0 = init.flatten()
            try:
                res = minimize(objective_smooth, x0, args=(beta,), method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                pts = res.x.reshape(-1, 2)
                update_best(pts)
            except Exception:
                pass
    
    # Phase 2: Nelder-Mead polish from best and top candidates
    if best_points is not None:
        # Polish the best
        if time.time() < deadline - 15.0:
            x0 = best_points.flatten()
            try:
                res = minimize(objective_nonsmooth, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-11})
                pts = res.x.reshape(-1, 2)
                update_best(pts)
            except Exception:
                pass
        
        # Try polishing from slightly perturbed best
        if time.time() < deadline - 10.0:
            for _ in range(3):
                perturbed = best_points + rng.random((N, 2)) * 0.01
                x0 = perturbed.flatten()
                try:
                    res = minimize(objective_nonsmooth, x0, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11})
                    pts = res.x.reshape(-1, 2)
                    update_best(pts)
                except Exception:
                    pass
    
    # Phase 3: Final high-precision polish
    if best_points is not None and time.time() < deadline - 3.0:
        x0 = best_points.flatten()
        try:
            res = minimize(objective_nonsmooth, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-13,
                                    'adaptive': True})
            pts = res.x.reshape(-1, 2)
            update_best(pts)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = initializations[0]
        best_score = score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
