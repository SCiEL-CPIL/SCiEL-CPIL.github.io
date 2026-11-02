import math
import numpy as np
import time
from itertools import permutations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# The 11 cells of the minimal 3x3 hexagonal lattice block (axial coordinates)
# These form a hexagon of "radius" 2 in axial coordinates
CELLS = [
    (0, 0),
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    (1, 1), (-1, 2), (-2, 1), (2, -1)
]

def axial_to_cartesian(axial_list):
    """Convert axial coordinates to Cartesian coordinates for unit hexagons."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def calc_enclosing_hexagon_side(pts, center, beta):
    """
    Calculate the side length of the smallest enclosing regular hexagon
    with center, rotation beta, that contains all points.
    
    Returns (S, D) where S is the side length and D is the apothem distance.
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    cx, cy = center
    for nx, ny in normals:
        for px, py in pts:
            proj = abs((px - cx) * nx + (py - cy) * ny)
            if proj > max_proj:
                max_proj = proj
    
    D = max_proj
    S = (D + APOTHEM) / APOTHEM
    return S, D

def optimize_center_and_rotation(pts):
    """
    For a given set of points, find the optimal center and rotation
    that minimizes the enclosing hexagon side length.
    """
    best_S = float('inf')
    best_center = None
    best_beta = 0.0
    
    # The problem has 6-fold symmetry, so beta in [0, pi/6) is sufficient
    # Coarse grid search over beta
    n_beta = 36
    for i in range(n_beta):
        beta = i * (math.pi / 6.0) / n_beta
        
        # For a given beta, find optimal center via linear programming or median
        # The optimal center is the center of the smallest enclosing hexagon
        # For a hexagon with given orientation, the optimal center minimizes
        # the maximum projection in all 6 normal directions
        # This is equivalent to finding the Chebyshev center
        
        # Use the fact that for a given set of points and fixed normals,
        # the optimal center can be found by solving a small LP
        # Or we can use the centroid as a good initial guess and refine
        
        # Simple approach: try several candidate centers
        # 1. Centroid
        centroid = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
        
        # 2. Try to find optimal center by binary search / optimization
        # For simplicity, use Nelder-Mead on the center
        
        from scipy.optimize import minimize
        
        def obj_center(cx_cy):
            cx_c, cy_c = cx_cy
            S, _ = calc_enclosing_hexagon_side(pts, (cx_c, cy_c), beta)
            return S
        
        try:
            res = minimize(obj_center, centroid, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_center = (res.x[0], res.x[1])
                best_beta = beta
        except Exception:
            pass
    
    # Refine beta around the best found
    if best_S < float('inf'):
        from scipy.optimize import minimize_scalar
        
        def obj_beta(b):
            # Re-optimize center for this beta
            centroid = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
            
            def obj_center(cx_cy):
                cx_c, cy_c = cx_cy
                S, _ = calc_enclosing_hexagon_side(pts, (cx_c, cy_c), b)
                return S
            
            try:
                res = minimize(obj_center, centroid, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                return res.fun
            except Exception:
                return float('inf')
        
        try:
            res = minimize_scalar(obj_beta, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 200, 'xatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                # Re-find center
                centroid = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
                
                def obj_center(cx_cy):
                    cx_c, cy_c = cx_cy
                    S, _ = calc_enclosing_hexagon_side(pts, (cx_c, cy_c), best_beta)
                    return S
                
                res2 = minimize(obj_center, centroid, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                best_center = (res2.x[0], res2.x[1])
        except Exception:
            pass
    
    return best_S, best_center, best_beta

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # The 11 cells already form a valid packing (they are disjoint unit hexagons
    # on the hexagonal lattice). The key insight is that the minimal enclosing
    # hexagon for these 11 cells has a known structure.
    
    # For the 3x3 hexagonal block (radius 2 in axial coords), the minimal
    # enclosing regular hexagon has side length related to the diameter.
    
    # Let's compute the optimal enclosing hexagon for the standard 3x3 block
    pts = axial_to_cartesian(CELLS)
    
    # Find optimal center and rotation
    best_S, best_center, best_beta = optimize_center_and_rotation(pts)
    
    # If optimization failed, use a reasonable default
    if best_center is None or best_S == float('inf'):
        best_center = (0.0, 0.0)
        best_beta = 0.0
        best_S = calc_enclosing_hexagon_side(pts, best_center, best_beta)[0]
    
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Compute the actual side length with safety margin
    S_final, D_final = calc_enclosing_hexagon_side(pts, best_center, best_beta)
    
    # Add tiny safety margin
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recompute side length for scaled points
    S_scaled, _ = calc_enclosing_hexagon_side(pts_scaled, best_center, best_beta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
