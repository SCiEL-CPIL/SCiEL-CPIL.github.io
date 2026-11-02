"""Heilbronn triangle problem: maximize minimum triangle area for 11 points.

Strategy:
1. Greedy initialization: Start with vertices and iteratively add points to maximize
   the minimum distance to existing points (approximating max-min area).
2. Vectorized evaluation: Compute all triangle areas at once for speed.
3. Nelder-Mead refinement: Use scipy.optimize.minimize with Nelder-Mead method
   on a soft-min objective function to polish the configuration.
4. Multiple restarts with different seeds if time permits.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 118.0  # safety margin before 120s limit

def _all_triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points.
    Returns an array of shape (n, n, n) where areas[i,j,k] is the area of triangle (i,j,k).
    Note: areas[i,j,k] == areas[j,i,k] etc. We only need upper triangle for min, but 
    full matrix is easier to vectorize.
    """
    n = len(pts)
    # Vectorized computation of cross products
    # pts: (n, 2)
    # We want area(i, j, k) = 0.5 * | (pts[j]-pts[i]) x (pts[k]-pts[i]) |
    
    # Expand dimensions to broadcast
    # p_i: (n, 1, 2)
    # p_j: (1, n, 2)
    # p_k: (1, 1, n, 2) -> This is too big (11^3 is small, but let's be careful)
    
    # Actually, 11 points -> 165 triangles. 
    # Looping over 165 triangles is fast enough in Python if we vectorize the point ops?
    # No, let's just use the explicit loop for clarity and correctness, 
    # but optimize the inner calculation.
    
    # For n=11, number of triangles is 11*10*9/6 = 165.
    # A pure python loop of 165 iterations is negligible.
    # However, the current program's _min_triangle_area is the bottleneck?
    # Let's check the speed. 165 iterations is < 1ms.
    # So the bottleneck is the number of evaluations, not the cost per evaluation.
    
    # Let's stick to a clean, fast implementation.
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                areas.append(0.5 * cross)
    return np.array(areas)

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    areas = _all_triangle_areas(pts)
    return np.min(areas)

def _soft_min_objective(pts, beta=100.0):
    """
    Objective function for optimization.
    We want to maximize the minimum area.
    This is equivalent to minimizing -min(areas).
    The min function is non-differentiable. We use a soft-min approximation:
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    As beta -> inf, soft_min -> min.
    
    We minimize -soft_min(areas) = 1/beta * log(sum(exp(-beta * areas)))
    Wait, we want to maximize min(areas).
    Let f(pts) = min(areas).
    We want to maximize f.
    Let g(pts) = -f(pts). We want to minimize g.
    g is non-differentiable.
    
    Approximation:
    min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    So -min(x) ≈ 1/beta * log(sum(exp(-beta * x)))
    
    We minimize J(pts) = 1/beta * log(sum(exp(-beta * areas(pts))))
    """
    areas = _all_triangle_areas(pts)
    # To avoid overflow in exp, shift by max
    # But areas are positive and small.
    # exp(-beta * area) can underflow if beta*area is large.
    # If beta=100 and area=0.01, exp(-1) ~ 0.36.
    # If area=0.001, exp(-0.1) ~ 0.9.
    # If area=0.1, exp(-10) ~ 0.00004.
    # This is fine.
    
    # For numerical stability, we can use the log-sum-exp trick, but here
    # we are summing exp(-beta * x).
    # Let's just compute it directly.
    log_sum = np.log(np.sum(np.exp(-beta * areas)))
    return log_sum / beta

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        # Clamp y to [0, min(sqrt(3)*x, sqrt(3)*(1-x))]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        # Clamp x to [0, 1]
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _greedy_init(rng):
    """
    Greedy initialization:
    1. Start with the 3 vertices of the triangle.
    2. Iteratively add a point that maximizes the minimum distance to existing points.
    3. This creates a well-distributed set.
    4. Finally, apply a few steps of local refinement to adjust positions.
    """
    # Vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, S3H])
    
    pts = [v0, v1, v2]
    
    for _ in range(N - 3):
        best_pt = None
        best_min_dist = -1
        
        # Sample candidate points
        # Use a grid or random samples
        num_samples = 500
        candidates = []
        for _ in range(num_samples):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            candidates.append([x, y])
        
        # Also add midpoints of existing edges and some strategic points
        # Midpoints of sides
        candidates.append([(v0[0]+v1[0])/2, (v0[1]+v1[1])/2])
        candidates.append([(v1[0]+v2[0])/2, (v1[1]+v2[1])/2])
        candidates.append([(v2[0]+v0[0])/2, (v2[1]+v0[1])/2])
        # Centroid
        candidates.append([(v0[0]+v1[0]+v2[0])/3, (v0[1]+v1[1]+v2[1])/3])
        
        for c in candidates:
            c = np.array(c)
            min_dist = float('inf')
            for p in pts:
                d = np.linalg.norm(c - p)
                if d < min_dist:
                    min_dist = d
            
            # We want to maximize the min distance
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pt = c
        
        if best_pt is not None:
            pts.append(best_pt)
    
    return np.array(pts)

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Phase 1: Greedy Initialization
    initial_pts = _greedy_init(rng)
    initial_pts = _project_to_triangle(initial_pts)
    current_area = _min_triangle_area(initial_pts)
    
    if current_area > best_area:
        best_area = current_area
        best_pts = initial_pts.copy()
    
    # Phase 2: Local Refinement using Nelder-Mead
    # We optimize the soft-min objective
    # Since Nelder-Mead doesn't handle constraints, we rely on projection or penalty.
    # A common trick: if a point goes outside, the objective becomes very bad?
    # Or we can just project after each step? Nelder-Mead doesn't allow custom step callbacks easily.
    # Alternative: Use SLSQP with bounds? But bounds are complex for triangle.
    # Let's stick to Nelder-Mead and hope it stays inside, or use a penalty term.
    
    # Penalty approach:
    # If any point is outside, add a large penalty to the objective.
    
    def penalized_objective(pts):
        # Check if inside
        for i in range(len(pts)):
            x, y = pts[i]
            if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9 or y > np.sqrt(3) * x + 1e-9 or y > np.sqrt(3) * (1 - x) + 1e-9:
                # Return a large value (bad)
                return 1000.0 + _soft_min_objective(pts, beta=50.0)
        
        return _soft_min_objective(pts, beta=50.0)
    
    # Start from the greedy init
    x0 = initial_pts.flatten()
    
    # Nelder-Mead parameters
    options = {'maxiter': 1000, 'fatol': 1e-6, 'xatol': 1e-6, 'disp': False}
    
    try:
        result = minimize(penalized_objective, x0, method='Nelder-Mead', options=options)
        
        if result.success or result.fun < 1000.0:
            refined_pts = result.x.reshape(-1, 2)
            refined_pts = _project_to_triangle(refined_pts)
            refined_area = _min_triangle_area(refined_pts)
            
            if refined_area > best_area:
                best_area = refined_area
                best_pts = refined_pts.copy()
    except Exception as e:
        pass # Ignore optimization errors
    
    # Phase 3: Additional restarts if time permits
    # The greedy init is deterministic with a fixed seed, so we need to vary it.
    # We can add random jitter to the greedy init or use different random seeds for the sampling.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Create a varied initial configuration
        # Jitter the greedy points
        base_pts = _greedy_init(rng)
        # Add small random noise
        jitter = rng.gauss(0, 0.05)
        varied_pts = base_pts + rng.uniform(-jitter, jitter, size=(N, 2))
        varied_pts = _project_to_triangle(varied_pts)
        
        current_area = _min_triangle_area(varied_pts)
        if current_area > best_area:
            best_area = current_area
            best_pts = varied_pts.copy()
        
        # Refine
        x0 = varied_pts.flatten()
        try:
            # Use a slightly higher beta for sharper min approximation in later runs?
            # Or just same.
            result = minimize(penalized_objective, x0, method='Nelder-Mead', options=options)
            
            if result.success or result.fun < 1000.0:
                refined_pts = result.x.reshape(-1, 2)
                refined_pts = _project_to_triangle(refined_pts)
                refined_area = _min_triangle_area(refined_pts)
                
                if refined_area > best_area:
                    best_area = refined_area
                    best_pts = refined_pts.copy()
        except Exception:
            pass
    
    # Final check and return
    if best_pts is None:
        best_pts = _greedy_init(rng)
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts
