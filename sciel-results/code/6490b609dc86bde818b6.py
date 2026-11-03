"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _penalty_outside(points):
    """
    Calculate penalty for points outside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    Returns a large positive value if any point is outside.
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += (0 - y) * 100
        val_left = np.sqrt(3) * x - y
        if val_left < 0:
            penalty += (-val_left) * 100
        val_right = np.sqrt(3) * (1 - x) - y
        if val_right < 0:
            penalty += (-val_right) * 100
    return penalty

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    Uses a 'rubber band' model with a hexagonal grid topology and global affine optimization.
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    
    # Define the "ideal" hexagonal grid topology for 11 points.
    # We use a hexagonal packing pattern that fits well in an equilateral triangle.
    # Coordinates in a normalized hex grid (where side length is 1):
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 1 point
    # Total: 10 points. We need 11.
    # Let's adjust: 
    # Row 0: 4 points
    # Row 1: 3 points
    # Row 2: 3 points (middle one is the 11th?)
    # Or maybe a different topology.
    # Let's try a 3x4 grid-like structure but sheared.
    # Actually, let's stick to the hexagonal idea but ensure 11 points.
    # A common optimal configuration for 11 points involves points on the boundary.
    
    # Let's define a base hex grid with 11 points.
    # Hex grid points (in unit hex coordinates, will be scaled/affinely transformed):
    # Using axial coordinates (q, r)
    # Center at (0,0)
    # Radius 1: 7 points
    # Radius 2: 12 points (too many)
    # Let's take a subset.
    
    # Alternative: Use a 4-3-3-1 row structure?
    # Let's define the base points in a coordinate system where the triangle is (0,0), (1,0), (0,1) for simplicity, then transform.
    # Or better, define them in the target triangle's coordinate space directly as an initial guess, then optimize affine transform + perturbations.
    
    # Let's use a known good configuration:
    # 11 points:
    # 4 on bottom edge
    # 3 in next row
    # 2 in next row
    # 1 at top
    # That's 10. Where is the 11th?
    # Maybe 3 on bottom, 3 in next, 3 in next, 2 at top?
    
    # Let's try to generate a hex grid of 11 points.
    # Hex grid generation:
    # We'll generate points in a hex lattice and select 11 that fit best.
    # Or, we can define the 11 points as a perturbation of a 3x4 grid.
    
    # Let's use the following topology:
    # Row 0 (y=0): 4 points at x = 0.1, 0.3, 0.5, 0.7 (scaled)
    # Row 1 (y=h/3): 3 points at x = 0.2, 0.4, 0.6 (scaled)
    # Row 2 (y=2h/3): 3 points at x = 0.3, 0.5, 0.7 (scaled)
    # Row 3 (y=h): 1 point at x = 0.5
    # Total: 4+3+3+1 = 11.
    
    # Define these points in the unit triangle (0,0), (1,0), (0.5, h)
    # We need to ensure they are inside.
    
    base_points = []
    # Row 0: y=0
    for x in [0.1, 0.3, 0.5, 0.7]:
        base_points.append([x, 0.0])
    # Row 1: y=h/3
    y1 = h / 3.0
    for x in [0.2, 0.4, 0.6]:
        base_points.append([x, y1])
    # Row 2: y=2h/3
    y2 = 2 * h / 3.0
    for x in [0.3, 0.5, 0.7]:
        base_points.append([x, y2])
    # Row 3: y=h
    base_points.append([0.5, h])
    
    base_points = np.array(base_points)
    
    # Now, we will optimize an affine transformation of these base points, plus small local perturbations.
    # Affine transform: p' = A @ p + b
    # A is 2x2, b is 2x1. Total 6 parameters.
    # Plus 11*2 = 22 perturbations.
    # Total 28 parameters.
    
    # However, we can reduce the search space by fixing the affine transform to map the base points into the triangle,
    # and only optimizing the perturbations.
    
    # Let's define the objective function that takes perturbations and returns the negative min area.
    
    def objective(perturbations):
        pts = base_points + perturbations.reshape(11, 2)
        # Check if points are inside the triangle
        pen = _penalty_outside(pts)
        if pen > 1e-6:
            return pen * 100
        min_area = _min_triangle_area(pts)
        return -min_area
    
    # Initial perturbations: zero
    x0 = np.zeros(22)
    
    # Use differential evolution for global optimization
    # Bounds for perturbations: small, to keep points near the base grid
    bounds = [(-0.1, 0.1) for _ in range(22)]
    
    # Run differential evolution
    res = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-6, popsize=15)
    
    best_pts = base_points + res.x.reshape(11, 2)
    best_score = -res.fun
    
    # Refine with Nelder-Mead
    def objective_nm(perturbations):
        pts = base_points + perturbations.reshape(11, 2)
        pen = _penalty_outside(pts)
        if pen > 1e-6:
            return pen * 100
        min_area = _min_triangle_area(pts)
        return -min_area
    
    res2 = minimize(objective_nm, res.x, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if res2.fun < res.fun:
        best_pts = base_points + res2.x.reshape(11, 2)
        best_score = -res2.fun
    
    # Final check
    if _penalty_outside(best_pts) > 1e-6:
        # Fallback to base points if invalid
        best_pts = base_points
        best_score = _min_triangle_area(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
