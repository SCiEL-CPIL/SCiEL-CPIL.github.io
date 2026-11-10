# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the Heilbronn score: min_triangle_area / hull_area."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or a_min <= 0:
        return 0.0
    return a_min / a_hull

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    """Generate vertices of a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _initial_guess():
    """Create an initial configuration: regular 13-gon on boundary."""
    return _regular_polygon(13, radius=1.0)

def _local_optimize(points, max_iter=1000, tol=1e-8):
    """
    Local optimization using L-BFGS-B to maximize the minimum triangle area
    relative to the hull area. We use a penalty-based approach:
    maximize min_area - penalty * (hull_area - target_hull)
    But actually we want to maximize min_area/hull_area.
    
    A better approach: fix the hull area by scaling, then maximize min_area.
    Or use the ratio directly with a smooth approximation.
    
    Let's use: minimize -log(min_area) + log(hull_area)
    But log of min is not smooth. Instead, let's try:
    minimize -(min_area / hull_area) directly, but it's not smooth at min.
    
    Alternative: use a soft-min approximation or just optimize with the raw ratio
    using numerical gradients.
    """
    # Flatten points
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Check for degenerate cases
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = _score(pts)
        if s <= 0:
            return 1e10
        return -s
    
    # Initial evaluation
    f0 = objective(x0)
    
    # Use L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50})
        if result.success or result.fun < f0:
            best_x = result.x
        else:
            best_x = x0
    except Exception:
        best_x = x0
    
    return best_x.reshape(-1, 2)

def _multi_start_optimize(n_starts=50, max_iter=500, time_budget=580):
    """
    Multi-start local optimization.
    """
    start_time = time.time()
    best_points = None
    best_score = -1.0
    
    # Start with regular polygon
    initial = _initial_guess()
    current = _local_optimize(initial, max_iter=max_iter)
    current_score = _score(current)
    if current_score > best_score:
        best_score = current_score
        best_points = current.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return best_points
    
    # Try different initial configurations
    rng = np.random.default_rng(seed=123)
    
    for i in range(n_starts - 1):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 30:  # Leave 30s safety margin
            break
        
        # Create perturbed initial guess
        base = _regular_polygon(13, radius=1.0)
        # Add random perturbation
        perturbation = rng.normal(0, 0.1, (13, 2))
        initial = base + perturbation
        
        # Ensure points are in reasonable range
        initial = np.clip(initial, -2, 2)
        
        current = _local_optimize(initial, max_iter=max_iter)
        current_score = _score(current)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current.copy()
    
    return best_points

def _refine_with_fine_steps(points, num_refinements=3):
    """
    Additional refinement with smaller steps and more iterations.
    """
    start_time = time.time()
    current = points.copy()
    current_score = _score(current)
    
    for level in range(num_refinements):
        # Increase max_iter for finer convergence
        max_iter = 1000 + level * 500
        tol = 1e-9 * (10 ** (-level))
        
        refined = _local_optimize(current, max_iter=max_iter, tol=tol)
        refined_score = _score(refined)
        
        if refined_score > current_score:
            current = refined
            current_score = refined_score
        
        elapsed = time.time() - start_time
        if elapsed > 550:  # Safety margin
            break
    
    return current

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    
    # Phase 1: Multi-start local optimization
    best_points = _multi_start_optimize(n_starts=30, max_iter=300, time_budget=500)
    
    elapsed = time.time() - start_time
    if elapsed > 550 or best_points is None:
        # Fallback to regular polygon if optimization failed or timed out
        return _regular_polygon(13, radius=1.0)
    
    # Phase 2: Fine refinement
    refined_points = _refine_with_fine_steps(best_points, num_refinements=2)
    
    # Final check
    final_score = _score(refined_points)
    if final_score <= 0:
        return _regular_polygon(13, radius=1.0)
    
    return refined_points

# EVOLVE-BLOCK-END
