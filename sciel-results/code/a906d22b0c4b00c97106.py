"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import sys

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -50, 0)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum

def _convex_params_to_points(params):
    """
    Convert convex polygon parameters to points.
    params: array of shape (N,) representing turning angles (in radians) for each edge.
    We fix p_0 at origin and first edge direction to x-axis.
    The turning angles determine the shape of the convex polygon.
    """
    n = len(params)
    # Turning angles should be in (0, pi) for convexity
    # Clip to ensure convexity
    angles = np.clip(params, 0.05, np.pi - 0.05)
    
    # Edge directions: cumulative sum of turning angles
    # First edge direction is 0 (along x-axis)
    edge_dirs = np.zeros(n)
    for i in range(1, n):
        edge_dirs[i] = edge_dirs[i-1] + angles[i-1]
    
    # We need to determine edge lengths such that the polygon closes.
    # Use a parameterization where we fix the scale and solve for lengths.
    # For now, use equal edge lengths as a starting point, then optimize.
    # Actually, let's use a different parameterization: 
    # Use angles and a single scale factor, with edge lengths determined by closure.
    
    # Better approach: use the fact that for a convex polygon with given turning angles,
    # the edge lengths are determined up to a scale by the closure condition.
    # Let's use a simpler parameterization: represent points in polar coordinates
    # relative to the centroid, with sorted angles.
    
    # Alternative: use the "edge vector" representation
    # Let's use a more practical approach: parameterize by (angle_i, r_i) for each point
    # where angle_i is the polar angle and r_i is the distance from centroid.
    # But this doesn't guarantee convexity easily.
    
    # Most practical: use the fact that we can represent a convex polygon by its 
    # support function or by vertices in order. Let's use vertices in order with
    # the constraint that they form a convex polygon.
    
    # For this implementation, let's use a hybrid approach:
    # - Use angles from centroid (sorted) and radii
    # - This guarantees convex position if angles are sorted and radii are positive
    
    # Actually, let me use a cleaner parameterization:
    # params[0:N-1] = turning angles (N-1 values, last is determined by closure)
    # The Nth turning angle = 2*pi - sum(first N-1)
    # Then we need edge lengths. Use 1 scale parameter + relative lengths.
    
    # For simplicity and robustness, let's use the polar parameterization:
    # params[0:N] = (theta_0, r_0, theta_1, r_1, ..., theta_{N-1}, r_{N-1})
    # where theta_i are sorted angles and r_i > 0
    # This gives 2N parameters, but we can fix theta_0 = 0 and r_0 = 1 for gauge fixing.
    
    # Given the complexity, let's use a simpler approach:
    # Keep the original Cartesian parameterization but add a convexity penalty.
    # This is more robust and the inspirations suggest this is the right direction.
    
    # For now, fall back to Cartesian with a strong convexity enforcement
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _convexity_penalty(pts):
    """
    Compute a penalty for non-convex position.
    Returns 0 if all points are in convex position, else a positive penalty.
    """
    try:
        hull = ConvexHull(pts)
        hull_vertices = hull.vertices
        # If all points are on the hull, it's convex
        if len(hull_vertices) == N:
            return 0.0
        # Penalty: number of interior points
        return (N - len(hull_vertices)) * 10.0
    except Exception:
        return 100.0

def _objective(params, tau):
    """
    Objective function: minimize -soft_min(ratios) + convexity penalty.
    params: 22 values representing p_3 ... p_13.
    p_1 is fixed at (0,0), p_2 is fixed at (1,0).
    """
    pts = _convex_params_to_points(params)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _soft_min(ratios, tau)
    
    # Add convexity penalty
    pen = _convexity_penalty(pts)
    return -approx_min + pen

def _params_to_points(params):
    """Convert parameter vector to point array."""
    return _convex_params_to_points(params)

def _generate_convex_initial(rng):
    """
    Generate an initial configuration with points in convex position.
    Uses a regular polygon with perturbations.
    """
    # Generate angles for a convex polygon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Perturb angles slightly
    angles = base_angles + rng.uniform(-0.05, 0.05, N)
    # Sort to maintain order
    angles = np.sort(angles)
    
    # Generate radii (distance from centroid)
    radii = 1.0 + rng.normal(0, 0.05, N)
    radii = np.abs(radii) + 0.1  # Ensure positive
    
    # Compute points
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Center at origin
    pts -= np.mean(pts, axis=0)
    
    # Transform: p[0] -> (0,0), p[1] -> (1,0)
    # First, find the two closest points to use as reference
    # For now, use the first two points
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return rng.uniform(-1, 1, 22)
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    
    return pts[2:].flatten()

def _generate_known_heilbronn_initial():
    """
    Generate an initial configuration based on known Heilbronn configurations.
    For N=13, use a configuration close to the known optimal.
    """
    # Known good configuration for Heilbronn's problem
    # Use a regular 13-gon as a starting point
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Transform: p[0] -> (0,0), p[1] -> (1,0)
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return np.zeros(22)
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    
    return pts[2:].flatten()

def _local_optimize(params, tau, maxiter=500):
    """Local optimization using L-BFGS-B."""
    try:
        res = minimize(
            _objective,
            params,
            args=(tau,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        if res.fun < _objective(params, tau):
            return res.x.copy()
        return params.copy()
    except Exception:
        return params.copy()

def _coordinate_descent(params, tau, max_iter=100, step_size=0.01):
    """
    Coordinate descent optimization for the Heilbronn problem.
    Optimizes one coordinate at a time.
    """
    current = params.copy()
    current_score = -_objective(current, tau)
    
    for _ in range(max_iter):
        improved = False
        for i in range(len(current)):
            # Try increasing and decreasing
            for delta in [step_size, -step_size]:
                trial = current.copy()
                trial[i] += delta
                trial_score = -_objective(trial, tau)
                if trial_score > current_score:
                    current = trial
                    current_score = trial_score
                    improved = True
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points in convex position
    starts = []
    
    # Known Heilbronn configuration
    starts.append(_generate_known_heilbronn_initial())
    
    # Convex polygon starts with varying perturbations
    for i in range(10):
        if (time.time() - start_time) > time_limit - 40.0:
            break
        p = _generate_convex_initial(rng)
        starts.append(p)
    
    # Phase 2: Multi-scale tau annealing with L-BFGS-B
    tau_schedule = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            try:
                remaining = time_limit - (time.time() - start_time)
                maxiter = min(800, max(100, int(remaining * 5)))
                
                starts[i] = _local_optimize(starts[i], tau, maxiter)
                
                # Coordinate descent polish
                if (time.time() - start_time) < time_limit - 12.0:
                    starts[i] = _coordinate_descent(starts[i], tau, max_iter=50, step_size=0.005)
                
                # Evaluate true score
                pts = _params_to_points(starts[i])
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = starts[i].copy()
                    
            except Exception:
                pass
    
    # Phase 3: Refinement with higher tau
    if best_params is not None and (time.time() - start_time) < time_limit - 10.0:
        for tau in [20000, 50000, 100000]:
            if (time.time() - start_time) > time_limit - 5.0:
                break
            best_params = _local_optimize(best_params, tau, 2000)
            best_params = _coordinate_descent(best_params, tau, max_iter=100, step_size=0.001)
            pts = _params_to_points(best_params)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
    
    # Construct final points
    if best_params is None:
        best_params = _generate_known_heilbronn_initial()
    
    final_pts = _params_to_points(best_params)
    
    return final_pts

# EVOLVE-BLOCK-END
