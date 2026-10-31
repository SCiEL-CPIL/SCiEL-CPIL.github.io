"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
# Vertices of the equilateral triangle with area 1 (unit-area triangle)
# Vertices: (0,0), (1,0), (0.5, h) where h = sqrt(3)/2
H = np.sqrt(3) / 2.0
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])
TRI_AREA = 0.5 * 1.0 * H  # = sqrt(3)/4

def in_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1.0 + 1e-9:
        return False
    # Check against edge from (0,0) to (0.5, H): line y = sqrt(3)*x
    if y > np.sqrt(3) * x + 1e-9:
        return False
    # Check against edge from (1,0) to (0.5, H): line y = -sqrt(3)*(x-1)
    if y > -np.sqrt(3) * (x - 1.0) + 1e-9:
        return False
    return True

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def penalty(points):
    """Penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        if not in_triangle(p):
            # Distance to triangle boundary (approximate)
            x, y = p
            # Distance to bottom edge y=0
            d1 = max(0, -y)
            # Distance to left edge y = sqrt(3)*x
            d2 = max(0, np.sqrt(3)*x - y)
            # Distance to right edge y = -sqrt(3)*(x-1)
            d3 = max(0, y + np.sqrt(3)*(x-1))
            pen += d1 + d2 + d3
    return pen

def objective(params):
    """Objective function for optimization. Returns negative min area (to maximize)."""
    points = params.reshape(-1, 2)
    pen = penalty(points)
    if pen > 1e-6:
        return -1e-10 + pen * 1000  # Strong penalty for being outside
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initial guess: spread points somewhat evenly
    # Use a few different initial configurations
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Start with a regular grid-like pattern
    # Place points in a structured way
    def make_initial_config(method='grid'):
        if method == 'grid':
            # Try to place points in a hexagonal-like pattern
            pts = []
            # Bottom row
            for i in range(4):
                x = 0.15 + i * 0.23
                y = 0.05
                pts.append([x, y])
            # Middle row
            for i in range(3):
                x = 0.25 + i * 0.25
                y = 0.25
                pts.append([x, y])
            # Top row
            for i in range(2):
                x = 0.35 + i * 0.3
                y = 0.45
                pts.append([x, y])
            # One more at top
            pts.append([0.5, 0.6])
            return np.array(pts[:11])
        elif method == 'random':
            pts = []
            while len(pts) < 11:
                u, v = rng.random(2)
                if u + v < 1.0:
                    x = u + 0.5 * v
                    y = H * v
                    pts.append([x, y])
            return np.array(pts)
        else:  # 'vertices'
            # Start with vertices and midpoints
            pts = [V0, V1, V2,
                   (V0+V1)/2, (V1+V2)/2, (V2+V0)/2,
                   (2*V0+V1+V2)/3, (V0+2*V1+V2)/3, (V0+V1+2*V2)/3,
                   (V0+V1+V2)/3]
            return np.array(pts[:11])
    
    methods = ['grid', 'random', 'vertices']
    
    for method in methods:
        if time.time() > deadline - 5.0:
            break
        
        init = make_initial_config(method)
        # Flatten for optimizer
        x0 = init.flatten()
        
        try:
            # Use differential evolution for global search
            bounds = []
            for i in range(11):
                bounds.append((0.0, 1.0))  # x in [0,1]
                bounds.append((0.0, H))   # y in [0, H]
            
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                polish=True,
                callback=None
            )
            
            if result.success or result.fun < objective(x0):
                pts = result.x.reshape(-1, 2)
                # Clamp to triangle
                for i in range(len(pts)):
                    x, y = pts[i]
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(H, y))
                    # Project into triangle if outside
                    if not in_triangle(pts[i]):
                        # Simple projection: move towards centroid
                        cx, cy = 0.5, H/3
                        t = 0.9
                        pts[i] = [x + t*(cx-x), y + t*(cy-y)]
                        if not in_triangle(pts[i]):
                            pts[i] = [cx, cy]
                
                score = min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception as e:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # Local refinement using L-BFGS-B
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            x0 = best_points.flatten()
            bounds = []
            for i in range(11):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, H))
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            pts = result.x.reshape(-1, 2)
            for i in range(len(pts)):
                x, y = pts[i]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(H, y))
                if not in_triangle(pts[i]):
                    cx, cy = 0.5, H/3
                    pts[i] = [cx, cy]
            
            score = min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception as e:
            pass
    
    if best_points is None:
        # Fallback: return initial grid config
        best_points = make_initial_config('grid')
    
    # Final validation: ensure all points are inside
    for i in range(len(best_points)):
        if not in_triangle(best_points[i]):
            best_points[i] = [0.5, H/3]
    
    return best_points
