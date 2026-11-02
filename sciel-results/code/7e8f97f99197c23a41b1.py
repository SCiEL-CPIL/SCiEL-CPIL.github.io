"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points.
    Returns the raw area (not normalized)."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

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

def _all_triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
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

def _soft_min_objective(x_flat, beta=50.0):
    """
    Smooth approximation of the negative minimum triangle area.
    Uses the soft-min function: -1/beta * log(sum(exp(-beta * a_i)))
    This is differentiable and allows gradient-based methods to work.
    """
    pts = x_flat.reshape(N, 2)
    
    # Check bounds
    if not _check_inside(pts, eps=1e-6):
        return 1e10
    
    areas = _all_triangle_areas(pts)
    
    # Soft min: approximates min(areas)
    # To avoid overflow/underflow, subtract max area (which is -min of negative areas)
    # Let's use the stable log-sum-exp trick for soft min.
    # soft_min(a) = -1/beta * log(sum(exp(-beta * a)))
    # = -1/beta * log(sum(exp(beta * (-a))))
    # Let z = -beta * areas
    # log(sum(exp(z))) = max(z) + log(sum(exp(z - max(z))))
    z = -beta * areas
    z_max = np.max(z)
    log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
    soft_min_val = -1.0 / beta * log_sum_exp
    
    return -soft_min_val # We want to minimize negative area, so maximize area

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _initial_guess(rng):
    """Generate an initial set of 11 points."""
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Strategy: Use Nelder-Mead on a smoothed objective function.
    # The smoothing parameter beta controls the sharpness of the minimum.
    # We start with a larger beta (smoother) and might refine with a smaller one if time permits.
    
    max_starts = 50 # Increase starts since NM is fast per iteration but we need good starts
    
    for trial in range(max_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate initial points
        pts = _initial_guess(rng)
        x0 = pts.flatten()
        
        # Run Nelder-Mead optimization on the soft-min objective
        # Nelder-Mead is derivative-free but benefits from a smooth landscape
        try:
            # Use a moderate beta for the main search
            beta = 20.0 
            result = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
        except Exception:
            continue
        
        opt_pts = result.x.reshape(N, 2)
        opt_pts = _project_to_triangle(opt_pts)
        
        if _check_inside(opt_pts):
            area = _min_triangle_area(opt_pts)
            if area > best_area:
                best_area = area
                best_pts = opt_pts.copy()
        
        # Also try the initial points
        init_area = _min_triangle_area(pts)
        if init_area > best_area:
            best_area = init_area
            best_pts = pts.copy()
            
        # If time permits, do a local polish with a higher beta (sharper min)
        # or a few iterations of L-BFGS-B on the sharp objective starting from this result
        if time.time() - start_time < DEADLINE - 10 and best_pts is not None:
             # Try a quick polish with L-BFGS-B on the original non-smooth objective
             # This can fine-tune the result found by the smooth search
             try:
                 def _sharp_objective(x_flat):
                     pts = x_flat.reshape(N, 2)
                     if not _check_inside(pts, eps=1e-6):
                         return 1e10
                     return -_min_triangle_area(pts)
                 
                 # Start from the best point found so far
                 x_start = best_pts.flatten()
                 res_polish = minimize(
                     _sharp_objective,
                     x_start,
                     method='L-BFGS-B',
                     options={'maxiter': 100, 'maxfun': 500}
                 )
                 polish_pts = res_polish.x.reshape(N, 2)
                 polish_pts = _project_to_triangle(polish_pts)
                 if _check_inside(polish_pts):
                     polish_area = _min_triangle_area(polish_pts)
                     if polish_area > best_area:
                         best_area = polish_area
                         best_pts = polish_pts.copy()
             except Exception:
                 pass

    if best_pts is None:
        # Fallback: just return random points
        pts = _initial_guess(rng)
        return pts
    
    return best_pts
