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

def smooth_min_area(points, alpha=100.0):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    This provides gradients for SLSQP.
    """
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Shift to avoid overflow/underflow in exp
    a_max = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-alpha * (areas - a_max))))
    smooth_min = a_max - (1.0 / alpha) * log_sum
    return smooth_min

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    best_score = -1.0
    best_points = None
    
    # Strategy: Multi-start SLSQP on full 26D space with diverse initializations.
    # This breaks the symmetry constraint of the previous approach and explores
    # different "islands" (basins of attraction) in the solution space.
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    # 1. Regular polygon (centered)
    # 2. Regular polygon + center
    # 3. Random points in unit disk
    # 4. Perturbed regular polygon
    
    inits = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(poly.flatten())
    
    # Regular hexagon + 6 inner + 1 center (symmetric, but full 26D start)
    angles_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = angles_outer + np.pi / 6
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    center = np.array([[0.0, 0.0]])
    sym_pts = np.vstack([outer_pts, inner_pts, center])
    inits.append(sym_pts.flatten())
    
    # Random points in unit disk
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
        
    # Perturbed regular polygon
    for _ in range(5):
        pert = rng.normal(0, 0.1, (N, 2))
        pts = poly + pert
        inits.append(pts.flatten())
        
    def objective_full(x):
        pts = x.reshape(-1, 2)
        # Normalize to unit area to keep points bounded and score consistent
        a_hull = convex_hull_area(pts)
        if a_hull < 1e-10:
            return 1e6
        pts_norm = pts / np.sqrt(a_hull)
        # We want to maximize the smooth min area of the normalized points
        # Since score is scale invariant, we can just maximize the smooth min area
        # of the normalized points.
        sm = smooth_min_area(pts_norm, alpha=50.0)
        return -sm
    
    # Optimize each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        try:
            # Use SLSQP for gradient-based optimization
            # Bounds: keep points within a reasonable box, e.g., [-2, 2]
            bounds = [(-2, 2)] * (2 * N)
            
            res = minimize(objective_full, init, method='SLSQP', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < objective_full(init):
                pts = res.x.reshape(-1, 2)
                s = score(pts)
                
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass

    # If we found a good point, do a final polish with Nelder-Mead on the best point
    # to handle any non-smoothness or boundary issues
    if best_points is not None and time.time() < deadline - 5.0:
        def objective_nms(x):
            pts = x.reshape(-1, 2)
            return -score(pts)
            
        try:
            x0 = best_points.flatten()
            res = minimize(objective_nms, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            pts = res.x.reshape(-1, 2)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
