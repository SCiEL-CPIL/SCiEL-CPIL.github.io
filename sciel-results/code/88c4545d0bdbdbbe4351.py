"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
SQRT3 = 3 ** 0.5
TRI_AREA = SQRT3 / 4.0  # area of the unit-side equilateral triangle

def _min_area(points):
    """Compute the minimum area among all C(N,3) triangles."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                a = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
                if a < min_a:
                    min_a = a
    return min_a

def _min_area_vec(pts_flat):
    """Vectorized min area for optimization (negative for maximization)."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    # Compute all triangle areas vectorized
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                if a < min_a:
                    min_a = a
    return -min_a

def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle."""
    result = np.array(pts, dtype=float)
    for i in range(len(result)):
        x, y = result[i]
        # Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
        # Constraints: y >= 0, y <= sqrt(3)*x, y <= -sqrt(3)*(x-1)
        # i.e. y >= 0, sqrt(3)*x - y >= 0, -sqrt(3)*x + y + sqrt(3) >= 0
        if y < 0:
            y = 0.0
        if y > SQRT3 * x:
            # project onto line y = sqrt(3)*x
            t = y / (1 + 3)  # not quite right, let's do proper projection
            # Actually, let's just clamp
            x = max(0.0, min(1.0, x))
            y = min(SQRT3 * x, SQRT3 * (1 - x), SQRT3 / 2)
            # Re-project properly
            # The three edges:
            # Edge 1: y = 0 (bottom)
            # Edge 2: y = sqrt(3)*x (left)
            # Edge 3: y = -sqrt(3)*(x - 1) = sqrt(3)*(1-x) (right)
            # Find closest point on the triangle
            # Simple approach: check which constraint is violated and project
            pass
        # Proper projection
        x, y = result[i]
        # Check three half-planes:
        # h1 = y >= 0
        # h2 = sqrt(3)*x - y >= 0  => y <= sqrt(3)*x
        # h3 = -sqrt(3)*x + y + sqrt(3) >= 0 => y <= sqrt(3)*(1-x)
        if y < 0:
            # Project onto y=0, but also respect other constraints
            # Closest point on y=0 within the triangle
            x_new = min(max(x, 0.0), 1.0)
            y_new = 0.0
            # Check if (x_new, 0) is in triangle: need 0 <= sqrt(3)*x_new and 0 <= sqrt(3)*(1-x_new)
            # Both true for x_new in [0,1]
            result[i] = [x_new, y_new]
        elif y > SQRT3 * x:
            # Project onto line y = sqrt(3)*x
            # Line: sqrt(3)*x - y = 0
            # Normal: (sqrt(3), -1), norm = 2
            # Distance: (sqrt(3)*x - y) / 2
            # Projection: (x,y) - (sqrt(3)*x - y)/2 * (sqrt(3), -1)
            d = (SQRT3 * x - y) / 2.0
            x_new = x - d * SQRT3
            y_new = y + d
            # Clamp to triangle bounds
            x_new = max(0.0, min(1.0, x_new))
            y_new = max(0.0, min(SQRT3 * x_new, SQRT3 * (1 - x_new), SQRT3 / 2))
            result[i] = [x_new, y_new]
        elif y > SQRT3 * (1 - x):
            # Project onto line y = sqrt(3)*(1-x) => sqrt(3)*x + y - sqrt(3) = 0
            # Normal: (sqrt(3), 1), norm = 2
            d = (SQRT3 * x + y - SQRT3) / 2.0
            x_new = x - d * SQRT3
            y_new = y - d
            x_new = max(0.0, min(1.0, x_new))
            y_new = max(0.0, min(SQRT3 * x_new, SQRT3 * (1 - x_new), SQRT3 / 2))
            result[i] = [x_new, y_new]
    return result

def _initial_config():
    """A good starting configuration based on known Heilbronn-type arrangements."""
    # Use a symmetric arrangement: some points near vertices, some near edges, some in center
    # For 11 points, a common approach is to use a hexagonal-like packing or known optimal configs
    # Let's use a configuration inspired by known results:
    # Place points at vertices of a smaller equilateral triangle pattern
    
    pts = []
    # 3 vertices of the big triangle (slightly inset to allow room)
    inset = 0.02
    pts.append([inset, inset * SQRT3])  # near (0,0) but inside
    pts.append([1 - inset, inset * SQRT3])  # near (1,0)
    pts.append([0.5, SQRT3/2 - inset * SQRT3])  # near top
    
    # 3 points near midpoints of edges
    pts.append([0.25, SQRT3 * 0.25])  # left edge midpoint
    pts.append([0.75, SQRT3 * 0.25])  # right edge midpoint  
    pts.append([0.5, 0.0])  # bottom edge midpoint

    # 2 points in the lower interior
    pts.append([0.35, SQRT3 * 0.15])
    pts.append([0.65, SQRT3 * 0.15])
    
    # 2 more interior points
    pts.append([0.5, SQRT3 * 0.3])
    pts.append([0.4, SQRT3 * 0.4])
    
    # 1 more
    pts.append([0.6, SQRT3 * 0.35])
    
    return np.array(pts[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Start with a reasonable initial configuration
    best_pts = _initial_config()
    best_pts = _project_to_triangle(best_pts)
    best_score = _min_area(best_pts)
    
    # Try a few different initial configurations
    initial_configs = []
    
    # Config 1: symmetric with center point
    c1 = np.array([
        [0.02, 0.035], [0.98, 0.035], [0.5, 0.84],
        [0.25, 0.433], [0.75, 0.433], [0.5, 0.02],
        [0.35, 0.26], [0.65, 0.26], [0.5, 0.52],
        [0.42, 0.65], [0.58, 0.65]
    ])
    initial_configs.append(c1)
    
    # Config 2: more spread out
    c2 = np.array([
        [0.05, 0.09], [0.95, 0.09], [0.5, 0.8],
        [0.2, 0.35], [0.8, 0.35], [0.5, 0.05],
        [0.3, 0.2], [0.7, 0.2], [0.5, 0.45],
        [0.4, 0.6], [0.6, 0.6]
    ])
    initial_configs.append(c2)
    
    # Config 3: hexagonal-like
    c3 = np.array([
        [0.1, 0.17], [0.9, 0.17], [0.5, 0.75],
        [0.3, 0.52], [0.7, 0.52], [0.5, 0.1],
        [0.25, 0.3], [0.75, 0.3], [0.5, 0.4],
        [0.4, 0.65], [0.6, 0.65]
    ])
    initial_configs.append(c3)
    
    # Add the initial config
    initial_configs.append(best_pts)
    
    # Evaluate all initial configs
    for cfg in initial_configs:
        cfg = _project_to_triangle(cfg)
        score = _min_area(cfg)
        if score > best_score:
            best_score = score
            best_pts = cfg.copy()
    
    # Optimize using Nelder-Mead
    def objective(x):
        pts = x.reshape(-1, 2)
        pts = _project_to_triangle(pts)
        return -_min_area(pts)
    
    # Try optimizing from the best initial config
    x0 = best_pts.flatten()
    
    iterations = 0
    while time.time() < deadline - 5.0:
        # Run Nelder-Mead
        try:
            res = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            x_opt = res.x
            pts_opt = _project_to_triangle(x_opt.reshape(-1, 2))
            score_opt = _min_area(pts_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
                x0 = best_pts.flatten()
            else:
                # If no improvement, break
                break
        except Exception:
            break
        
        iterations += 1
        if iterations >= 10:
            break
    
    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    
    # Verify all points are inside the triangle
    for i in range(N):
        x, y = best_pts[i]
        if y < 0 or y > SQRT3 * x or y > SQRT3 * (1 - x):
            # Re-project
            best_pts[i] = _project_to_triangle(best_pts[i:i+1])[0]
    
    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Points:", pts)
    print("Min area:", _min_area(pts))
    print("Score (min area / triangle area):", _min_area(pts) / TRI_AREA)
