"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

# EVOLVE-BLOCK-START

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def _all_triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points. Returns a list of areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    areas = _all_triangle_areas(pts)
    return min(areas) if areas else 0.0

def _inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the reference triangle."""
    x, y = p
    # Edge AB: y >= 0
    if y < -tol:
        return False
    # Edge BC: line from (1,0) to (0.5, sqrt(3)/2). y <= sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x) + tol:
        return False
    # Edge CA: line from (0.5, sqrt(3)/2) to (0,0). y <= sqrt(3)*x
    if y > np.sqrt(3) * x + tol:
        return False
    return True

def _clamp_to_triangle(p):
    """Clamp a point to be inside the triangle."""
    x, y = p
    # First, clip to the bounding box
    if x < 0: x = 0
    if x > 1: x = 1
    if y < 0: y = 0
    if y > np.sqrt(3)/2: y = np.sqrt(3)/2
    
    # Now project onto the triangle if outside
    # Constraint 1: y <= sqrt(3)*(1-x)  =>  sqrt(3)*x + y <= sqrt(3)
    val1 = np.sqrt(3)*x + y - np.sqrt(3)
    if val1 > 0:
        # Project onto line sqrt(3)*x + y = sqrt(3)
        # Normal vector n = (sqrt(3), 1), |n|^2 = 4
        # p_new = p - (val1/4) * n
        x -= (val1 / 4) * np.sqrt(3)
        y -= (val1 / 4) * 1
        
    # Constraint 2: y <= sqrt(3)*x  =>  -sqrt(3)*x + y <= 0  =>  sqrt(3)*x - y >= 0
    val2 = y - np.sqrt(3)*x
    if val2 > 0:
        # Project onto line sqrt(3)*x - y = 0
        # Normal vector n = (-sqrt(3), 1) for the half-plane y <= sqrt(3)*x? 
        # Let's use the line equation: -sqrt(3)*x + y = 0. Normal n = (-sqrt(3), 1).
        # Distance d = (-sqrt(3)*x + y) / 2 = val2 / 2
        # p_new = p - d * n = p - (val2/2)*(-sqrt(3), 1) = p + (val2/2)*(sqrt(3), -1)
        x += (val2 / 2) * np.sqrt(3)
        y -= (val2 / 2) * 1
        
    # Final safety clip
    if x < 0: x = 0
    if x > 1: x = 1
    if y < 0: y = 0
    if y > np.sqrt(3)/2: y = np.sqrt(3)/2
    
    return np.array([x, y])

def _penalty(pts):
    """Penalty for points outside the triangle."""
    pen = 0
    for p in pts:
        if not _inside_triangle(p):
            x, y = p
            # Approximate distance
            d1 = max(0, -y)
            d2 = max(0, y - np.sqrt(3)*(1-x))
            d3 = max(0, y - np.sqrt(3)*x)
            pen += max(d1, d2, d3)
    return pen

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s
    
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Multiple random restarts with Nelder-Mead
    n_starts = 15
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate random initial configuration using barycentric coordinates
        pts = []
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            # P = u*V1 + v*V2 + (1-u-v)*V0
            x = u * 1.0 + v * 0.5
            y = v * np.sqrt(3) / 2
            pts.append([x, y])
        pts = np.array(pts)
        
        # Optimize using Nelder-Mead
        def objective(x):
            p = x.reshape(N, 2)
            min_a = _min_area(p)
            pen = _penalty(p)
            return -min_a + 100 * pen
        
        try:
            res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            final_pts = res.x.reshape(N, 2)
            # Clamp to ensure validity
            final_pts = np.array([_clamp_to_triangle(p) for p in final_pts])
            score = _min_area(final_pts)
            if _penalty(final_pts) < 1e-6 and score > best_score:
                best_score = score
                best_points = final_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: random points
        pts = []
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u * 1.0 + v * 0.5
            y = v * np.sqrt(3) / 2
            pts.append([x, y])
        return np.array(pts)
    
    current_best = best_points.copy()
    current_score = best_score
    
    # Strategy 2: Local search refinement
    # Use a finer-grained local search to improve the best solution found
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        step_size = 0.05
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < deadline - 2 and no_improve_count < max_no_improve:
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                # Try a few random perturbations
                for _ in range(5):
                    delta = rng.normal(0, step_size, 2)
                    new_pts = current_best.copy()
                    new_pts[i] = current_best[i] + delta
                    new_pts[i] = _clamp_to_triangle(new_pts[i])
                    
                    new_score = _min_area(new_pts)
                    if new_score > current_score + 1e-10:
                        current_best = new_pts
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                step_size *= 0.9
                if step_size < 1e-5:
                    step_size = 1e-5
    
    return current_best

# EVOLVE-BLOCK-END
