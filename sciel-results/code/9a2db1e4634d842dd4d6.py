"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
DEADLINE = 140.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
DEADLINE = 140.0
START_TIME = time.time()

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -eps:
        return False
    if y > np.sqrt(3) * x + eps:
        return False
    if y > -np.sqrt(3) * (x - 1) + eps:
        return False
    return True

def _all_inside(points, eps=1e-9):
    return all(_inside_triangle(p, eps) for p in points)

def _apply_affine(base_points, params):
    """
    params = [scale, rotation, tx, ty]
    p' = scale * R(rotation) @ p + [tx, ty]
    """
    scale, rot, tx, ty = params
    R = np.array([[np.cos(rot), -np.sin(rot)],
                  [np.sin(rot),  np.cos(rot)]])
    M = scale * R
    transformed = (base_points @ M.T) + np.array([tx, ty])
    return transformed

def _objective(params, base_points):
    """Negative minimum triangle area with penalty for points outside."""
    points = _apply_affine(base_points, params)
    
    # Penalty for points outside
    penalty = 0.0
    for p in points:
        x, y = p
        d1 = max(0.0, -y)
        d2 = max(0.0, y - np.sqrt(3) * x)
        d3 = max(0.0, y + np.sqrt(3) * (x - 1))
        penalty += d1**2 + d2**2 + d3**2
    
    if penalty > 1e-10:
        return 1.0 + penalty
    
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    global START_TIME, DEADLINE
    
    # Base pattern: 11 points in a triangular lattice
    # Using a hexagonal/triangular arrangement
    # Rows: 4, 3, 2, 1, 1 = 11
    # Or better: use a symmetric pattern
    
    # Let's define base points in a coordinate system where they form a nice pattern
    # Triangular lattice with spacing 1
    base = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [1.5, np.sqrt(3)/2],
        [2.5, np.sqrt(3)/2],
        [1.0, np.sqrt(3)],
        [2.0, np.sqrt(3)],
        [0.0, np.sqrt(3)],
        [3.0, np.sqrt(3)],
    ])
    
    # Center the base pattern
    base_center = base.mean(axis=0)
    base_centered = base - base_center
    
    best_points = None
    best_score = -1e10
    
    # Initial guesses: [scale, rotation, tx, ty]
    # The triangle has area sqrt(3)/4, base pattern spans about 3 x sqrt(3)
    # Scale should be around 0.2-0.3 to fit
    initial_guesses = [
        [0.25, 0.0, 0.5, np.sqrt(3)/6],
        [0.2, 0.0, 0.5, np.sqrt(3)/6],
        [0.3, 0.0, 0.5, np.sqrt(3)/6],
        [0.25, 0.1, 0.5, np.sqrt(3)/6],
        [0.25, -0.1, 0.5, np.sqrt(3)/6],
        [0.25, 0.2, 0.5, np.sqrt(3)/6],
        [0.25, -0.2, 0.5, np.sqrt(3)/6],
        [0.22, 0.0, 0.5, np.sqrt(3)/6],
        [0.28, 0.0, 0.5, np.sqrt(3)/6],
    ]
    
    for x0 in initial_guesses:
        if time.time() > DEADLINE - 10:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(base_centered,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < 1.0:  # No penalty
                points = _apply_affine(base_centered, result.x)
                if _all_inside(points, eps=1e-8):
                    score = _min_triangle_area(points)
                    if score > best_score:
                        best_score = score
                        best_points = points
        except Exception:
            continue
    
    # Fallback if optimization failed
    if best_points is None:
        # Use a simple triangular grid
        h = np.sqrt(3) / 2
        points = []
        # Row 0: y=0.05, x from 0.1 to 0.9
        for i in range(4):
            points.append([0.15 + i * 0.2, 0.05])
        # Row 1: y=0.15
        for i in range(3):
            points.append([0.25 + i * 0.2, 0.15])
        # Row 2: y=0.25
        for i in range(2):
            points.append([0.35 + i * 0.2, 0.25])
        # Row 3: y=0.35
        points.append([0.45, 0.35])
        # Row 4: y=0.4
        points.append([0.5, 0.4])
        best_points = np.array(points)
        
        if not _all_inside(best_points):
            # Scale and center
            cx, cy = best_points.mean(axis=0)
            best_points = (best_points - [cx, cy]) * 0.8 + [0.5, h/3]
            if not _all_inside(best_points):
                best_points = (best_points - [cx, cy]) * 0.5 + [0.5, h/3]
    
    return best_points
# EVOLVE-BLOCK-END
