# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                areas[i, j, k] = area
    return areas

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the Heilbronn score: A_min / A_hull.
    To make the optimization smoother and scale-invariant, we normalize 
    the points to have unit hull area and then maximize the minimum triangle area.
    """
    area_hull = _hull_area(points)
    if area_hull < 1e-10:
        return -1e10 # Penalty for degenerate hull
    
    # Scale points to have unit hull area
    scale = 1.0 / np.sqrt(area_hull)
    scaled_points = points * scale
    
    # Compute min triangle area on scaled points
    areas = _triangle_areas(scaled_points)
    min_area = np.min(areas)
    
    # Return the min area (which is A_min / sqrt(A_hull) * ... wait)
    # A_min_scaled = A_min_original * scale^2 = A_min_original / A_hull_original
    # So maximizing A_min_scaled is equivalent to maximizing A_min / A_hull.
    return min_area

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, leave margin
    
    # Initial guess: Hexagonal lattice
    # A hexagonal lattice is a good starting point for Heilbronn problems.
    # For n=13, we can take a hexagon of side length 2 (which has 19 points) 
    # and remove 6 points, or use a different configuration.
    # Let's use a simple hexagonal packing of 13 points.
    # Center (0,0), 6 at distance 1, 6 at distance 2 (but only some).
    # Actually, a known good config for n=13 is a perturbed hexagonal grid.
    
    # Let's generate a hexagonal grid of points and pick 13.
    # Or simpler: start with a random perturbation of a regular polygon?
    # No, hexagonal is better.
    
    # Hexagonal coordinates
    # Row 0: 1 point
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 4 points
    # Row 4: 3 points
    # Total: 13 points.
    
    points = []
    # Center at (0,0)
    # Let's define the hex grid manually for 13 points.
    # A "hexagon" of side 2 has 19 points. We need 13.
    # Let's use the 13 points from the center out to the first ring (7 points) 
    # and then 6 points from the second ring.
    
    # Ring 0: (0,0)
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points at distance 1, angles 0, 60, 120, 180, 240, 300
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
        
    # Ring 2: We need 6 more points.
    # The second ring has 12 points. Let's pick every other one to maintain symmetry?
    # Or just pick 6 consecutive ones?
    # Let's pick points at angles 30, 90, 150, 210, 270, 330 (offset by 30 deg)
    # Distance is 2 * cos(pi/6) = sqrt(3) for the "flat" hex grid?
    # In a hex grid, the distance to the second ring points depends on the direction.
    # Let's just use a regular dodecagon for the outer 6 points?
    # Or better, use the actual hex grid coordinates.
    
    # Hex grid coordinates for Ring 2 (distance ~1.732 to 2.0)
    # Let's just use a regular 12-gon and pick 6 alternating vertices?
    # That might not be optimal.
    
    # Alternative: Use a random start and optimize. The optimizer should fix the geometry.
    # But a good start helps.
    # Let's use the 13 points of a hexagonal lattice of side length 2, 
    # specifically the center, the 6 neighbors, and 6 of the 12 next-neighbors.
    
    # Let's just use a random perturbation of a circle? No, hex is better.
    # Let's use the 13 points from the optimal n=13 solution if I can recall it?
    # I don't have it memorized.
    
    # Let's try a "star" shape: Center + 12 points on a circle.
    # This is often a good starting point for local search.
    points = [[0.0, 0.0]]
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points.append([np.cos(angle), np.sin(angle)])
    
    points = np.array(points)
    
    # Add small random noise to break symmetry
    rng = np.random.default_rng(seed=42)
    points += rng.normal(0, 0.01, points.shape)
    
    best_points = points
    best_score = _score(points)
    
    # Optimization
    # We minimize -score
    def objective(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    
    # Use Nelder-Mead for robustness
    # Options: maxiter, tol
    # We have ~600s. Nelder-Mead can be slow.
    # Let's try a few restarts with different initial points.
    
    # Initial point 1: Circle
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    if res.success or res.fun < -best_score:
        best_points = res.x.reshape(-1, 2)
        best_score = -res.fun
    
    if time.time() > deadline - 60:
        return best_points
    
    # Initial point 2: Hexagonal
    points_hex = []
    points_hex.append([0.0, 0.0])
    for i in range(6):
        angle = i * np.pi / 3
        points_hex.append([np.cos(angle), np.sin(angle)])
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3
        # Distance for hex grid ring 2 is 2 * cos(pi/6) = sqrt(3) ~ 1.732
        # Let's use 1.732
        points_hex.append([1.732 * np.cos(angle), 1.732 * np.sin(angle)])
    
    points_hex = np.array(points_hex)
    points_hex += rng.normal(0, 0.01, points_hex.shape)
    
    x0_hex = points_hex.flatten()
    res2 = minimize(objective, x0_hex, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    if res2.success or res2.fun < -best_score:
        best_points = res2.x.reshape(-1, 2)
        best_score = -res2.fun
    
    if time.time() > deadline - 30:
        return best_points
    
    # Refine the best point found
    x0_best = best_points.flatten()
    res3 = minimize(objective, x0_best, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    if res3.success or res3.fun < -best_score:
        best_points = res3.x.reshape(-1, 2)
        best_score = -res3.fun
        
    return best_points

# EVOLVE-BLOCK-END
