"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])
# Area of the triangle
TRI_AREA = 0.5 * np.abs((B[0]-A[0])*(C[1]-A[1]) - (C[0]-A[0])*(B[1]-A[1]))

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points from pts."""
    min_area = np.inf
    n = len(pts)
    # Use vectorized computation for speed
    # For each triplet (i, j, k), compute area
    # Area = 0.5 * |cross(Pj - Pi, Pk - Pi)|
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_vec(pts):
    """Vectorized computation of minimum triangle area."""
    n = len(pts)
    # Compute all cross products
    # For efficiency, we can use broadcasting
    # But for 11 points, 11 choose 3 = 165 triangles, which is small
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    return np.min(areas)

def _project_to_triangle(pt):
    """Project a point onto the triangle if it's outside."""
    # Check if point is inside the triangle using barycentric coordinates
    # or simply clip to the triangle boundaries
    
    x, y = pt
    
    # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # Left edge: from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x
    # Right edge: from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3)*(x-1)
    # Bottom edge: y = 0
    
    # If y < 0, project to y = 0
    if y < 0:
        y = 0
        # Adjust x to stay within [0, 1]
        x = np.clip(x, 0, 1)
        return np.array([x, y])
    
    # Check left edge: y <= sqrt(3)*x
    if y > np.sqrt(3)*x:
        # Project onto left edge
        # Line: y = sqrt(3)*x
        # Closest point on line to (x, y)
        # Minimize (x' - x)^2 + (y' - y)^2 subject to y' = sqrt(3)*x'
        # x' = (x + sqrt(3)*y) / 4
        # y' = sqrt(3)*(x + sqrt(3)*y) / 4
        x_proj = (x + np.sqrt(3)*y) / 4
        y_proj = np.sqrt(3) * x_proj
        # Clamp to segment [0, 0.5] in x
        if x_proj < 0:
            x_proj, y_proj = 0, 0
        elif x_proj > 0.5:
            x_proj, y_proj = 0.5, np.sqrt(3)/2
        return np.array([x_proj, y_proj])
    
    # Check right edge: y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    if y > np.sqrt(3)*(1-x):
        # Project onto right edge
        # Line: y = sqrt(3)*(1-x)
        # Minimize (x' - x)^2 + (y' - y)^2 subject to y' = sqrt(3)*(1-x')
        # Let t = 1 - x', then y' = sqrt(3)*t, x' = 1-t
        # Minimize (1-t-x)^2 + (sqrt(3)*t - y)^2
        # d/dt: 2(1-t-x)(-1) + 2(sqrt(3)*t - y)*sqrt(3) = 0
        # -(1-t-x) + 3t - sqrt(3)*y = 0
        # -1 + t + x + 3t - sqrt(3)*y = 0
        # 4t = 1 - x + sqrt(3)*y
        # t = (1 - x + sqrt(3)*y) / 4
        t = (1 - x + np.sqrt(3)*y) / 4
        x_proj = 1 - t
        y_proj = np.sqrt(3) * t
        # Clamp to segment [0.5, 1] in x
        if x_proj < 0.5:
            x_proj, y_proj = 0.5, np.sqrt(3)/2
        elif x_proj > 1:
            x_proj, y_proj = 1, 0
        return np.array([x_proj, y_proj])
    
    return np.array([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0
    best_points = None
    
    # Initialize with a reasonable starting configuration
    # Use a grid-like or random initialization
    
    def _objective(x):
        """Negative of the minimum triangle area (we maximize min area)."""
        pts = x.reshape(-1, 2)
        # Project points to ensure they're in the triangle
        pts = np.array([_project_to_triangle(p) for p in pts])
        min_area = _min_triangle_area(pts)
        return -min_area
    
    def _objective_grad(x):
        """Gradient of the objective (negative of min triangle area)."""
        pts = x.reshape(-1, 2)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        base_val = _objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (_objective(x_plus) - base_val) / eps
        return grad
    
    # Multi-start SLSQP optimization
    np.random.seed(42)
    
    n_starts = 50
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        # Generate initial points
        if s == 0:
            # Start with a regular grid-like pattern
            pts = []
            # Include vertices and some interior points
            pts.append([0.0, 0.0])
            pts.append([1.0, 0.0])
            pts.append([0.5, np.sqrt(3)/2])
            # Add interior points
            for i in range(1, 5):
                for j in range(1, 5):
                    if i + j < 6 and len(pts) < N:
                        # Barycentric coordinates
                        u = i / 5.0
                        v = j / 5.0
                        if u + v < 1:
                            x = u * 0.5 + v * 0.5
                            y = v * np.sqrt(3)/2
                            pts.append([x, y])
            while len(pts) < N:
                pts.append([0.5, 0.2])
            x0 = np.array(pts[:N]).flatten()
        else:
            # Random initialization
            pts = []
            while len(pts) < N:
                u, v = np.random.random(2)
                if u + v < 0.95:
                    x = u + 0.5 * v
                    y = np.sqrt(3)/2 * v
                    pts.append([x, y])
            x0 = np.array(pts).flatten()
        
        # Define constraints to keep points inside the triangle
        # The triangle is defined by:
        # y >= 0
        # y <= sqrt(3)*x
        # y <= sqrt(3)*(1-x)
        
        def _constraints(x):
            pts = x.reshape(-1, 2)
            cons = []
            for i in range(N):
                x_i, y_i = pts[i]
                # y_i >= 0
                cons.append(y_i)
                # y_i <= sqrt(3)*x_i => sqrt(3)*x_i - y_i >= 0
                cons.append(np.sqrt(3)*x_i - y_i)
                # y_i <= sqrt(3)*(1-x_i) => sqrt(3)*(1-x_i) - y_i >= 0
                cons.append(np.sqrt(3)*(1-x_i) - y_i)
            return cons
        
        constraints = [{'type': 'ineq', 'fun': _constraints}]
        
        bounds = [(0, 1)] * (2*N)
        
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or (hasattr(res, 'fun') and res.fun < 0):
                pts = res.x.reshape(-1, 2)
                # Project to ensure validity
                pts = np.array([_project_to_triangle(p) for p in pts])
                score = _min_triangle_area(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception as e:
            pass
    
    # If we didn't find any valid points, use a default
    if best_points is None:
        # Use a simple grid
        pts = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.3
            x = 0.5 + r * np.cos(angle)
            y = np.sqrt(3)/4 + r * np.sin(angle) * 0.5
            pts.append(_project_to_triangle([x, y]))
        best_points = np.array(pts)
    
    return best_points
