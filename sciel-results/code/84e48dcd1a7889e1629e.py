"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def inside_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    x, y = p
    # Barycentric coordinates
    # Area of triangle = sqrt(3)/4
    # For point (x,y), barycentric coords:
    # a = area of triangle (p, V1, V2) / total_area
    # b = area of triangle (V0, p, V2) / total_area
    # c = area of triangle (V0, V1, p) / total_area
    # Simpler: use half-plane tests
    # Edge V0-V1: y >= 0
    if y < -1e-9:
        return False
    # Edge V1-V2: line from (1,0) to (0.5, sqrt(3)/2)
    # Direction: (-0.5, sqrt(3)/2), normal pointing inward: (sqrt(3)/2, 0.5)
    # Point relative to V1: (x-1, y)
    # Dot product with inward normal should be >= 0
    if (x - 1.0) * (np.sqrt(3)/2) + y * 0.5 < -1e-9:
        return False
    # Edge V2-V0: line from (0.5, sqrt(3)/2) to (0,0)
    # Direction: (-0.5, -sqrt(3)/2), inward normal: (-sqrt(3)/2, 0.5)
    # Point relative to V2: (x-0.5, y-sqrt(3)/2)
    if (x - 0.5) * (-np.sqrt(3)/2) + (y - np.sqrt(3)/2) * 0.5 < -1e-9:
        return False
    return True

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(points_flat):
    """Objective function: negative of min triangle area (to minimize).
    Also adds a penalty if points go outside the triangle."""
    points = points_flat.reshape(N, 2)
    
    # Check if all points are inside the triangle
    for i in range(N):
        if not inside_triangle(points[i]):
            # Penalty: distance outside
            # Project to triangle
            # Simple penalty: large value
            return 1e6
    
    min_area = min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_value = -float('inf')
    
    # Initialize with a good starting configuration
    # Use a symmetric arrangement
    # Center + 10 points in two rings
    cx, cy = 0.5, np.sqrt(3)/6  # centroid
    
    # Try multiple random initializations and local optimizations
    rng = np.random.RandomState(42)
    
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial points
        if restart == 0:
            # Symmetric initial guess
            points = np.zeros((N, 2))
            points[0] = [cx, cy]
            # Inner ring: 5 points
            for i in range(5):
                angle = 2 * np.pi * i / 5 - np.pi/2
                r = 0.15
                points[i+1] = [cx + r*np.cos(angle), cy + r*np.sin(angle)]
            # Outer ring: 5 points
            for i in range(5):
                angle = 2 * np.pi * i / 5 - np.pi/2 + np.pi/5
                r = 0.3
                points[i+6] = [cx + r*np.cos(angle), cy + r*np.sin(angle)]
        else:
            # Random initialization inside triangle
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v <= 1.0:
                        p = [u + 0.5*v, np.sqrt(3)/2 * v]
                        if inside_triangle(np.array(p)):
                            points[i] = p
                            break
        
        # Local optimization using scipy
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if time.time() > deadline:
                break
            
            if result.success or result.fun < 0:  # Check if we got a valid result
                opt_points = result.x.reshape(N, 2)
                
                # Verify all points are inside
                valid = True
                for i in range(N):
                    if not inside_triangle(opt_points[i]):
                        valid = False
                        break
                
                if valid:
                    val = -result.fun
                    if val > best_value:
                        best_value = val
                        best_points = opt_points.copy()
        except Exception:
            continue
    
    # If no valid result found, return a default configuration
    if best_points is None:
        # Fallback: symmetric arrangement
        points = np.zeros((N, 2))
        points[0] = [cx, cy]
        for i in range(5):
            angle = 2 * np.pi * i / 5 - np.pi/2
            r = 0.15
            points[i+1] = [cx + r*np.cos(angle), cy + r*np.sin(angle)]
        for i in range(5):
            angle = 2 * np.pi * i / 5 - np.pi/2 + np.pi/5
            r = 0.3
            points[i+6] = [cx + r*np.cos(angle), cy + r*np.sin(angle)]
        best_points = points
    
    return best_points
