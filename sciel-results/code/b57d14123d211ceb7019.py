"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Represent points in Barycentric coordinates (lambda1, lambda2, lambda3) >= 0, sum=1.
   This naturally enforces the triangle constraint without projection.
2. Use a "Soft-Min" objective: -log(sum(exp(-areas/t))) or similar smooth approximation
   of the minimum area. This allows gradient-based optimization (L-BFGS-B).
3. Multi-start with diverse initializations:
   a. Structured symmetric initialization.
   b. "Largest Empty Triangle" heuristic: Start with vertices, iteratively place
      points to maximize the minimum distance to existing points, but biased towards
      the centroid of the largest empty triangle formed by existing points.
   c. Random barycentric points.
4. Anneal the temperature 't' of the soft-min to sharpen the focus on the minimum area.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    # bary shape (N, 3), sum along axis 1 is 1
    # cart = bary[:, 0:1]*V0 + bary[:, 1:2]*V1 + bary[:, 2:3]*V2
    # Since V0 is (0,0), we can simplify:
    # cart = bary[:, 1:2]*V1 + bary[:, 2:3]*V2
    return bary[:, 1:2] * V1 + bary[:, 2:3] * V2

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to Barycentric (N, 3)."""
    # Inverse of the transformation.
    # x = l1*1 + l2*0.5
    # y = l2*sqrt(3)/2
    # l2 = y / (sqrt(3)/2) = 2*y/sqrt(3)
    # l1 = x - l2*0.5
    # l0 = 1 - l1 - l2
    y = cart[:, 1]
    x = cart[:, 0]
    l2 = y / S3H
    l1 = x - 0.5 * l2
    l0 = 1.0 - l1 - l2
    return np.column_stack((l0, l1, l2))

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points in Cartesian coords."""
    # pts shape (N, 2)
    # Pre-compute indices
    idx = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    
    # Cross product magnitude / 2
    dx1 = pj - pi
    dx2 = pk - pi
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_area_cart(pts, t=0.01):
    """
    Compute a smooth approximation of the minimum area.
    We want to maximize the min area.
    Soft-min is usually defined for minima.
    Let A_i be the areas.
    min(A_i) approx -t * log(sum(exp(-A_i/t)))
    We want to maximize this, so we minimize -soft_min.
    """
    idx = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    
    dx1 = pj - pi
    dx2 = pk - pi
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Numerical stability for log-sum-exp
    min_a = np.min(areas)
    # If t is very small, this approximates min(areas)
    # soft_min = -t * log(sum(exp(-(areas - min_a)/t))) - min_a
    # Actually, simpler:
    # log_sum_exp(-areas/t) = log(sum(exp(-areas/t)))
    # To avoid overflow, subtract max(-areas/t) which is -min(areas)/t
    exp_terms = np.exp(-(areas - min_a) / t)
    sum_exp = np.sum(exp_terms)
    soft_min = min_a - t * np.log(sum_exp)
    return soft_min

def _objective_bary(bary_flat, t):
    """
    Objective function for L-BFGS-B.
    Input: flat array of barycentric coords (3*N,)
    Returns: -soft_min_area (to be minimized)
    """
    bary = bary_flat.reshape(N, 3)
    # Enforce non-negativity via penalty if needed, but L-BFGS-B has bounds
    # We use bounds [0, 1] for each bary coordinate.
    # The sum constraint is handled by the fact that we only optimize 2 coords?
    # No, L-BFGS-B works on unconstrained variables within bounds.
    # It is easier to parameterize with 2 variables per point (l1, l2) and set l0 = 1 - l1 - l2.
    # However, the bounds on l1, l2 are coupled.
    # Alternative: Optimize in Cartesian with penalty for being outside?
    # Or Optimize in Barycentric with bounds [0,1] and ignore the sum constraint?
    # If we ignore sum constraint, the point might not be in the plane.
    # Better: Parameterize by (u, v) in [0,1] for each point, where bary = (1-u-v, u, v).
    # This ensures sum=1 and non-negativity if u,v >=0 and u+v <=1.
    # Bounds for u,v: [0,1]. Constraint u+v <= 1 is not handled by simple bounds.
    # We can use a penalty for u+v > 1.
    
    # Let's stick to the 2-parameter representation: bary = (1-u-v, u, v)
    # Input flat is (2*N,)
    uv = bary_flat.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    l0 = 1.0 - u - v
    l1 = u
    l2 = v
    
    # Penalty for l0 < 0 (i.e., u+v > 1)
    penalty = 0.0
    neg_l0 = np.clip(-l0, 0, None)
    penalty = 100.0 * np.sum(neg_l0**2)
    
    bary = np.column_stack((l0, l1, l2))
    # Clip to ensure numerical stability in conversion if slightly negative due to float error
    bary = np.clip(bary, 0, 1)
    # Renormalize to sum to 1 just in case
    bary_sum = bary.sum(axis=1, keepdims=True)
    bary = bary / bary_sum
    
    cart = _bary_to_cart(bary)
    area = _soft_min_area_cart(cart, t)
    return penalty - area

def _local_refine_cart(pts_cart, t_init=0.05, t_final=0.001, max_iter=50):
    """
    Refine Cartesian points using L-BFGS-B on the (u,v) parameters.
    """
    bary = _cart_to_bary(pts_cart)
    # Convert bary (l0, l1, l2) to (u, v) where u=l1, v=l2
    uv = bary[:, 1:3].flatten()
    
    # Bounds for u, v are [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    best_obj = np.inf
    best_uv = uv.copy()
    
    # Annealing loop
    for i in range(max_iter):
        t = t_init * (t_final / t_init) ** (i / max_iter)
        
        res = minimize(
            _objective_bary, 
            best_uv, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 10, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        # Check if we improved
        if res.fun < best_obj:
            best_obj = res.fun
            best_uv = res.x
            
    # Convert back to Cartesian
    uv_best = best_uv.reshape(N, 2)
    l0 = 1.0 - uv_best[:, 0] - uv_best[:, 1]
    l1 = uv_best[:, 0]
    l2 = uv_best[:, 1]
    bary_best = np.column_stack((l0, l1, l2))
    bary_best = np.clip(bary_best, 0, 1)
    bary_best /= bary_best.sum(axis=1, keepdims=True)
    cart_best = _bary_to_cart(bary_best)
    
    return cart_best

def _largest_empty_triangle_init(rng):
    """
    Heuristic initialization:
    1. Start with the 3 vertices.
    2. Repeatedly find the "largest empty triangle" formed by existing points.
    3. Place the new point at the centroid of this triangle, or slightly perturbed.
    4. This tends to spread points out.
    """
    pts = [V0.copy(), V1.copy(), V2.copy()]
    
    for _ in range(N - 3):
        # Find largest area triangle among existing points
        max_area = -1
        best_idx = None
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                for k in range(j+1, len(pts)):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    # Area
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    if area > max_area:
                        max_area = area
                        best_idx = (i, j, k)
        
        if best_idx is None:
            break
            
        i, j, k = best_idx
        centroid = (pts[i] + pts[j] + pts[k]) / 3.0
        
        # Add a small random perturbation to avoid symmetry traps
        angle = rng.random() * 2 * np.pi
        dist = rng.random() * 0.05
        perturb = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        new_pt = centroid + perturb
        
        # Project to triangle if needed
        # Simple projection: clip to bounds
        x, y = new_pt
        if y < 0: y = 0
        if y > S3H: y = S3H
        if x < 0: x = 0
        if x > 1: x = 1
        # Check left edge y < sqrt(3)x
        if y > np.sqrt(3) * x:
            # Project onto line y = sqrt(3)x
            # This is a rough projection
            y = np.sqrt(3) * x
        # Check right edge y < sqrt(3)(1-x)
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x)
            
        pts.append(np.array([x, y]))
        
    return np.array(pts)

def _structured_init():
    """Symmetric structured initialization."""
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def _random_init(rng):
    """Random barycentric initialization."""
    bary = rng.random((N, 3))
    bary /= bary.sum(axis=1, keepdims=True)
    return _bary_to_cart(bary)

def _check_inside(pts, eps=1e-9):
    for p in pts:
        x, y = p
        if y < -eps or x < -eps or x > 1 + eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points slightly outside back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0: y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            # Project onto the boundary
            # If y > sqrt(3)*x, project to (y/sqrt(3), y) is not right, project to line
            # Line 1: y = sqrt(3)x => x = y/sqrt(3)
            # Line 2: y = sqrt(3)(1-x) => x = 1 - y/sqrt(3)
            if y > np.sqrt(3) * x:
                x = y / np.sqrt(3)
            else:
                x = 1 - y / np.sqrt(3)
        if x < 0: x = 0
        if x > 1: x = 1
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1
    
    # List of initializers
    initializers = []
    
    # 1. Structured
    initializers.append(_structured_init())
    
    # 2. Largest Empty Triangle Heuristic (multiple random restarts)
    for _ in range(5):
        initializers.append(_largest_empty_triangle_init(rng))
        
    # 3. Random
    for _ in range(5):
        initializers.append(_random_init(rng))
        
    # Sort initializers by initial score (descending) to prioritize good starts
    scored_inits = []
    for init_pts in initializers:
        init_pts = _project_to_triangle(init_pts)
        if not _check_inside(init_pts):
            continue
        score = _min_triangle_area_cart(init_pts)
        scored_inits.append((score, init_pts))
        
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Try refining the top initializations
    for score, init_pts in scored_inits:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        if score > best_score:
            best_pts = init_pts
            best_score = score
            
        # Refine using L-BFGS-B with annealing
        try:
            refined_pts = _local_refine_cart(init_pts)
            refined_pts = _project_to_triangle(refined_pts)
            if _check_inside(refined_pts):
                refined_score = _min_triangle_area_cart(refined_pts)
                if refined_score > best_score:
                    best_pts = refined_pts
                    best_score = refined_score
        except Exception:
            pass

    # If no valid points found (should not happen), fallback
    if best_pts is None:
        best_pts = _structured_init()
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area_cart(pts)}")
