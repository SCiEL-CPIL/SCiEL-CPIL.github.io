import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - S3 * x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _feasibility_obj(points, alpha):
    """
    Objective for feasibility check: 
    Minimize sum of squared violations of A_k >= alpha, plus boundary penalty.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Violation: (alpha - area)^2 if area < alpha, else 0
    # Using max(0, alpha - area)
    viol = np.maximum(0, alpha - areas)
    obj = np.sum(viol**2)
    
    # Boundary penalty
    pen = _penalty(points)
    # Weight penalty heavily to keep points inside
    obj += 1000.0 * pen
    
    return obj

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_population(rng, count=1):
    """
    Generate initial valid configurations.
    """
    configs = []
    for _ in range(count):
        pts = []
        for _ in range(N):
            pts.append(_random_valid_point(rng))
        configs.append(np.array(pts))
    return configs

def _check_feasibility(alpha, start_points, time_limit, rng):
    """
    Check if there exists a configuration with min area >= alpha.
    Returns (is_feasible, best_points, min_violation)
    """
    start_time = time.time()
    best_val = np.inf
    best_pts = start_points.copy()
    
    # Try a few random restarts if we have time, or just use the provided start
    # For speed, we primarily rely on the warm start from the previous alpha
    # But if the warm start fails, we try 2 random restarts
    
    attempts = [start_points]
    # Add 2 random restarts
    for _ in range(2):
        attempts.append(_initial_population(rng, 1)[0])
    
    for init_pts in attempts:
        if time.time() - start_time > time_limit:
            break
        
        def _obj(flat_pts):
            pts = flat_pts.reshape(-1, 2)
            return _feasibility_obj(pts, alpha)
        
        try:
            res = minimize(
                _obj, 
                init_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_pts = res.x.reshape(-1, 2)
                
                # If we found a very good solution, we can stop early
                if best_val < 1e-8:
                    return True, best_pts, best_val
                    
        except Exception:
            pass
            
    # Feasible if violation is very small
    # We also need to check if the points are actually inside (penalty should be ~0)
    pen = _penalty(best_pts)
    if best_val < 1e-5 and pen < 1e-8:
        return True, best_pts, best_val
    else:
        return False, best_pts, best_val

def construct():
    start_time = time.time()
    timeout = 120.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial guess for upper bound
    # Total area is ~0.433. 11 points. 
    # A rough upper bound is Area / (N * something). 
    # Let's start with a conservative hi and expand if needed, or just set a high hi.
    # The score 20.67 corresponds to min_area ~ 20.67 / (1/0.433) ? 
    # Wait, the task says "The evaluator measures the raw value". 
    # "The notebook uses an equilateral triangle and divides by its area."
    # So if raw min area is A, score = A / (sqrt(3)/4).
    # Score 20.67 => A = 20.67 * 0.433 / 1 ? No, "divides by its area".
    # If the triangle area is 1, then score = min_area.
    # The unit-area triangle has area 1. The vertices are (0,0), (1,0), (0.5, sqrt(3)/2).
    # Area of this triangle is 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4 ≈ 0.433.
    # If the evaluator divides by the area of the triangle, then Score = Min_Area / 0.433.
    # So Min_Area = Score * 0.433.
    # For Score 20.67, Min_Area ≈ 8.95. This is impossible as the triangle area is 0.433.
    
    # Re-reading: "Place 11 points on or inside a unit-area triangle".
    # "The notebook uses an equilateral triangle and divides by its area."
    # If the triangle used in the notebook has area 1, then the coordinates must be scaled.
    # But the interface says: "equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)".
    # This specific triangle has area sqrt(3)/4.
    # If the evaluator calculates score = min_area / area_of_triangle, then:
    # Score = min_area / (sqrt(3)/4).
    # If Score is 20.67, then min_area = 20.67 * 0.433 ≈ 8.95. Still impossible.
    
    # Let's check the seed program score. 20.6695.
    # Maybe the "unit-area triangle" means the triangle in the problem statement has area 1?
    # If the triangle in the problem has area 1, then the side length s satisfies s^2 * sqrt(3)/4 = 1 => s = sqrt(4/sqrt(3)) ≈ 1.519.
    # But the interface explicitly gives vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    # This triangle has area sqrt(3)/4.
    
    # Perhaps the score is just the min area, and the "divides by its area" refers to a normalization in the *description* of the problem source, but the evaluator just takes the raw min area?
    # If Score = Min_Area, then Min_Area = 20.67. Impossible.
    
    # Let's look at the "Inspiration 1" score: 20.6695.
    # If the max possible min area for 11 points in a triangle of area 0.433 is small, the score must be scaled.
    # Common scaling: Score = Min_Area / (Area_of_Triangle / N) ? Or Min_Area / Area_of_Triangle?
    # If Score = Min_Area / Area_Triangle, and Score ~ 20, then Min_Area ~ 8.9.
    
    # Let's assume the evaluator computes:
    # score = min_area / (sqrt(3)/4)
    # If the current program gets 20.67, it means min_area is around 8.95? No.
    
    # Let's re-read carefully: "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # This implies the "value" is what the evaluator calculates.
    # If the value is 20.67, and it's a "raw value", it's likely the min area itself?
    # No, 20.67 is too large for an area in a unit-ish triangle.
    
    # Maybe the triangle is (0,0), (1,0), (0,1)? Area 0.5.
    # The interface says: (0,0), (1,0), (1/2, sqrt(3)/2).
    
    # Let's assume the score IS the min area, and my intuition about the magnitude is wrong?
    # No, 11 points in a triangle of area 0.433.
    # Even if they are vertices of a grid, the min area is tiny.
    # Max min area for N points in a triangle of Area A is roughly A / (c * N^2).
    # For N=11, A=0.433. c ~ 1-2.
    # Min area ~ 0.433 / 121 ~ 0.0035.
    # If Score = Min_Area, Score ~ 0.0035.
    # But the score is 20.67.
    
    # Therefore, Score = Min_Area / Normalization.
    # If Score = 20.67 and Min_Area ~ 0.0035, then Normalization ~ 0.00017.
    # This doesn't match "divides by its area" (0.433).
    
    # Let's look at the code of the current program.
    # It returns `best_points`.
    # The evaluator calculates the score.
    # If I assume the evaluator calculates `min_area / (sqrt(3)/4)`, then:
    # Score = min_area / 0.433.
    # If Score = 20.67, min_area = 8.95. Impossible.
    
    # What if the triangle in the evaluator is different?
    # "unit-area triangle". Area = 1.
    # If the evaluator uses a triangle of Area 1, but the interface asks for points in (0,0)-(1,0)-(0.5, H), then the evaluator might scale the points?
    # Or maybe the "unit-area" description is just context, and the evaluator uses the specific triangle provided in the interface.
    
    # Let's assume the standard interpretation for these tasks:
    # The evaluator calculates the minimum area of triangles formed by the returned points.
    # Then it divides by the area of the containing triangle.
    # Containing Triangle Area = sqrt(3)/4.
    # Score = Min_Area / (sqrt(3)/4).
    # If the current score is 20.67, then Min_Area = 20.67 * 0.433 = 8.95.
    # This is geometrically impossible for 11 points in that triangle.
    
    # Is it possible the score is NOT min_area / area?
    # Maybe it's min_area * 1000?
    # If Min_Area ~ 0.02, Score ~ 20.
    # 0.02 / 0.433 = 0.046.
    # 0.02 * 1000 = 20.
    # This seems plausible.
    
    # Let's assume the goal is to maximize Min_Area.
    # The scaling doesn't change the optimal points.
    # So I will just maximize Min_Area.
    
    # My binary search bounds for Min_Area:
    # Lower: 0.0
    # Upper: 0.05 (Guess: 11 points, min area can be up to ~0.02-0.03?)
    # Let's set hi = 0.05.
    
    lo = 0.0
    hi = 0.05
    
    # Get an initial good point to use as warm start
    # Use a simple random start and polish it once to get a reasonable baseline
    init_pts = _initial_population(rng, 1)[0]
    
    # Quick polish to get a decent starting point for binary search
    def _min_smooth_obj(flat_pts):
        pts = flat_pts.reshape(-1, 2)
        Pi = pts[I]
        Pj = pts[J]
        Pk = pts[K]
        v1 = Pj - Pi
        v2 = Pk - Pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        beta = 50.0
        min_approx = -np.log(np.sum(np.exp(-beta * areas))) / beta
        pen = _penalty(pts)
        return -(min_approx - 1000.0 * pen)
        
    try:
        res = minimize(_min_smooth_obj, init_pts.flatten(), method='L-BFGS-B', options={'maxiter': 100})
        init_pts = res.x.reshape(-1, 2)
    except:
        pass
        
    current_best_pts = init_pts
    current_best_area = _min_area(init_pts)
    
    # Ensure hi is feasible? No, hi should be infeasible or just high.
    # Let's check if hi is feasible. If it is, increase hi.
    # If it's not, we have our range.
    
    # Binary Search
    # We have 120s.
    # Each feasibility check takes ~1-2s (L-BFGS-B with 22 vars, 165 constraints approx).
    # We can do ~50 iterations.
    
    time_limit_per_check = 1.5
    
    for _ in range(40):
        if time.time() - start_time > timeout - 10:
            break
            
        mid = (lo + hi) / 2.0
        
        # Use current_best_pts as warm start? 
        # Or a point close to the previous feasible solution?
        # We don't store the feasible points from the check, just the result.
        # Let's use current_best_pts as the start for the feasibility check.
        
        feasible, pts_found, val = _check_feasibility(mid, current_best_pts, time_limit_per_check, rng)
        
        if feasible:
            lo = mid
            # Update current best if the found points have a higher min area
            area_found = _min_area(pts_found)
            if area_found > current_best_area:
                current_best_area = area_found
                current_best_pts = pts_found
        else:
            hi = mid
            
    # Final Polish
    # Use the best alpha found (lo) and the points from the last feasible check
    # Or just polish current_best_pts with the smooth objective
    
    if time.time() - start_time < timeout - 5:
        try:
            res = minimize(
                _min_smooth_obj, 
                current_best_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            polished_pts = res.x.reshape(-1, 2)
            polished_area = _min_area(polished_pts)
            
            if polished_area > current_best_area:
                current_best_area = polished_area
                current_best_pts = polished_pts
        except Exception:
            pass
            
    # Final validity check
    x = current_best_pts[:, 0]
    y = current_best_pts[:, 1]
    
    y = np.maximum(y, 0)
    
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
        
    y = np.maximum(y, 0)
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
        
    current_best_pts[:, 0] = x
    current_best_pts[:, 1] = y
    
    return current_best_pts

construct()
