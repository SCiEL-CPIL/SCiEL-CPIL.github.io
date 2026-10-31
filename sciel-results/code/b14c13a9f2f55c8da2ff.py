import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for all 165 triangles
# This is much faster than np.indices every time
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
    Constraints:
    1. y >= 0
    2. y <= sqrt(3) * x
    3. y <= sqrt(3) * (1 - x)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    # 1. y < 0
    v1 = np.maximum(0, -y)
    # 2. y > sqrt(3) * x  =>  sqrt(3)*x - y < 0
    # The distance to the line ax+by+c=0 is |ax+by+c|/sqrt(a^2+b^2)
    # Line: sqrt(3)x - y = 0. Distance = (sqrt(3)x - y) / 2 if violated?
    # Actually, for penalty, just using the raw violation value is often sufficient if scaled correctly.
    # Let's use the perpendicular distance for a smoother penalty.
    # Line 1: y = 0. Dist = max(0, -y)
    # Line 2: y = sqrt(3)x => sqrt(3)x - y = 0. Norm = 2. Dist = max(0, (sqrt(3)x - y)/2) * 2? No, just the value.
    # Let's stick to the linear violation magnitude, it's robust.
    v2 = np.maximum(0, y - np.sqrt(3) * x)
    v3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(flat_points):
    points = flat_points.reshape(-1, 2)
    area = _min_area(points)
    pen = _penalty(points)
    # Maximize area, minimize penalty
    return -area + 100.0 * pen

def construct():
    start_time = time.time()
    timeout = 120.0
    
    # Bounds for DE: x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    best_result = None
    best_score = -np.inf
    
    def update_best(xk):
        nonlocal best_result, best_score
        pts = xk.reshape(-1, 2)
        # Only update if valid (low penalty)
        pen = _penalty(pts)
        if pen < 1e-6:
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_result = xk.copy()
        
    # 1. Differential Evolution
    # Use a callback to track the best result found so far
    def de_callback(xk, convergence):
        update_best(xk)
        if time.time() - start_time > timeout - 20:
            raise Exception("Timeout")

    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5, 
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            callback=de_callback
        )
        # Update best with the final result
        if res.fun < float('inf'):
            update_best(res.x)
    except Exception:
        pass

    # 2. Local Refinement with Nelder-Mead
    # If we have a good candidate, polish it.
    # Also, try a few random starts with Nelder-Mead to see if we find a better local optimum.
    
    candidates = []
    if best_result is not None:
        candidates.append(best_result)
    
    # Add a few random valid configurations
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    pts.append([x, y])
                    break
        candidates.append(np.array(pts).flatten())
    
    # Try to improve each candidate with Nelder-Mead
    for cand in candidates:
        if time.time() - start_time > timeout - 5:
            break
            
        try:
            # Nelder-Mead is robust for this
            res_local = minimize(
                _objective, 
                cand, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            update_best(res_local.x)
        except Exception:
            continue

    # 3. If still no valid result, return a simple valid one
    if best_result is None:
        # Fallback: Grid-like distribution
        # Just return a known valid configuration, e.g., vertices + center + random
        rng = np.random.default_rng(999)
        pts = []
        # Add vertices
        pts.append([0, 0])
        pts.append([1, 0])
        pts.append([0.5, H])
        # Add center
        pts.append([0.5, H/3])
        # Add 7 random points
        for _ in range(7):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    pts.append([x, y])
                    break
        best_result = np.array(pts).flatten()

    points = best_result.reshape(-1, 2)
    
    # Final check: project any slightly out-of-bounds points?
    # The penalty should have kept them in.
    # But just in case, do a hard projection?
    # No, the penalty is strong. If it's slightly out, the area might be valid but the evaluator might reject it?
    # The task says "inside or on".
    # Let's ensure validity by projecting.
    
    def project_point(p):
        x, y = p
        # If y < 0, y=0
        if y < 0:
            y = 0
        # If y > sqrt(3)*x, project to line
        if y > np.sqrt(3) * x:
            # Line: y = sqrt(3) x. Normal: (-sqrt(3), 1). 
            # Projection of (x,y) onto y=sqrt(3)x
            # t = (x*sqrt(3) + y*1) / (3+1) = (sqrt(3)x + y)/4
            # x_proj = t * 1? No.
            # Vector along line: (1, sqrt(3)). Unit: (1/2, sqrt(3)/2).
            # proj = dot(p, u) * u
            dot = x * 0.5 + y * (np.sqrt(3)/2)
            x = dot * 0.5
            y = dot * (np.sqrt(3)/2)
        # If y > sqrt(3)*(1-x), project to line
        if y > np.sqrt(3) * (1 - x):
            # Line: y = -sqrt(3)(x-1) = -sqrt(3)x + sqrt(3)
            # sqrt(3)x + y - sqrt(3) = 0
            # Normal: (sqrt(3), 1). Unit: (sqrt(3)/2, 1/2).
            # Direction of line: (1, -sqrt(3)). Unit: (1/2, -sqrt(3)/2).
            # Point on line: (1, 0).
            # Vector from (1,0) to (x,y): (x-1, y).
            # proj = (1,0) + dot(v, u) * u
            v = np.array([x-1, y])
            u = np.array([0.5, -np.sqrt(3)/2])
            dot = np.dot(v, u)
            x = 1 + dot * 0.5
            y = 0 + dot * (-np.sqrt(3)/2)
            
        return [x, y]

    for i in range(N):
        pts[i] = project_point(pts[i])

    return points
