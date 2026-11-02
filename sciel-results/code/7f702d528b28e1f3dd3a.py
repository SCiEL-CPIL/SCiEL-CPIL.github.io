import math
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Generate the 19 cells of a hexagon of radius 2
CELLS_19 = []
for q in range(-2, 3):
    for r in range(-2, 3):
        if abs(q + r) <= 2:
            CELLS_19.append((q, r))

def axial_to_cartesian(axial_list):
    """Convert axial coordinates to Cartesian coordinates for unit hexagons."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def calc_S_for_subset(pts):
    """
    Calculate the minimal enclosing hexagon side length for a set of points.
    Optimizes center and rotation.
    """
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Coarse grid search for beta
    n_beta = 12
    for i in range(n_beta):
        beta = i * (math.pi / 6.0) / n_beta
        
        # Normals for the 6 sides
        # Angles: beta + pi/6 + k*pi/3
        # We only need 3 unique axes (opposite normals give same projection range)
        axes = []
        for k in range(3):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            axes.append((math.cos(ang), math.sin(ang)))
        
        # Project points onto the 3 axes
        # For each axis, find min and max projection
        max_D = 0.0
        center_coords = []
        
        for nx, ny in axes:
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            min_p = np.min(projs)
            max_p = np.max(projs)
            D_axis = (max_p - min_p) / 2.0
            center_axis = (min_p + max_p) / 2.0
            center_coords.append((center_axis, (nx, ny)))
            if D_axis > max_D:
                max_D = D_axis
        
        # The optimal center is the intersection of the lines defined by the axis projections?
        # Actually, the center (cx, cy) must satisfy:
        # nx * cx + ny * cy = center_axis for each axis.
        # This is an overdetermined system (3 equations, 2 unknowns).
        # The optimal center minimizes the max error, but for the "width" calculation,
        # the D is determined by the spread.
        # Wait, the side length S is determined by the max projection distance from the center.
        # If we set the center to the "average" of the solutions, the max_D might increase.
        # However, for a convex polygon, the minimal enclosing hexagon center is unique.
        # For a set of points, the optimal center is the Chebyshev center.
        # A simple approximation: Use the centroid of the points?
        # Or solve the least squares for the center?
        
        # Let's use the least squares solution for the center (cx, cy)
        # A * [cx, cy]^T = b
        # Where A rows are (nx, ny) and b is center_axis
        
        A = np.array([(nx, ny) for _, (nx, ny) in center_coords])
        b = np.array([c for c, _ in center_coords])
        
        try:
            # Least squares
            res = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = res[0]
            center = (cx, cy)
            
            # Recalculate max_D with this center
            max_proj = 0.0
            for nx, ny in axes:
                # We need to check all 6 normals, but 3 axes is enough if we take abs
                # proj = (px - cx) * nx + (py - cy) * ny
                # max_abs_proj
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj_axis = np.max(np.abs(projs))
                if max_proj_axis > max_proj:
                    max_proj = max_proj_axis
            
            S = (max_proj + APOTHEM) / APOTHEM
            
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
                
        except Exception:
            pass

    # Refine beta
    if best_S < float('inf'):
        def obj_beta(b):
            # Re-compute center and S for this beta
            axes = []
            for k in range(3):
                ang = b + math.pi / 6.0 + k * math.pi / 3.0
                axes.append((math.cos(ang), math.sin(ang)))
            
            A = np.array([(nx, ny) for nx, ny in axes])
            
            # We need b for least squares.
            # b_k = (min_proj + max_proj) / 2
            b_vec = []
            for nx, ny in axes:
                projs = pts[:, 0] * nx + pts[:, 1] * ny
                min_p = np.min(projs)
                max_p = np.max(projs)
                b_vec.append((min_p + max_p) / 2.0)
            b = np.array(b_vec)
            
            try:
                res = np.linalg.lstsq(A, b, rcond=None)
                cx, cy = res[0]
                
                max_proj = 0.0
                for nx, ny in axes:
                    projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                    max_proj_axis = np.max(np.abs(projs))
                    if max_proj_axis > max_proj:
                        max_proj = max_proj_axis
                
                S = (max_proj + APOTHEM) / APOTHEM
                return S
            except Exception:
                return float('inf')
        
        try:
            res = minimize_scalar(obj_beta, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                # Re-find center
                axes = []
                for k in range(3):
                    ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
                    axes.append((math.cos(ang), math.sin(ang)))
                A = np.array([(nx, ny) for nx, ny in axes])
                b_vec = []
                for nx, ny in axes:
                    projs = pts[:, 0] * nx + pts[:, 1] * ny
                    min_p = np.min(projs)
                    max_p = np.max(projs)
                    b_vec.append((min_p + max_p) / 2.0)
                b = np.array(b_vec)
                res2 = np.linalg.lstsq(A, b, rcond=None)
                best_center = (res2[0][0], res2[0][1])
        except Exception:
            pass

    return best_S, best_center, best_beta

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_S_global = float('inf')
    best_subset = None
    best_center_global = None
    best_beta_global = 0.0

    # Iterate through all combinations of 11 cells from the 19
    for subset in combinations(CELLS_19, N):
        pts = axial_to_cartesian(list(subset))
        S, center, beta = calc_S_for_subset(pts)
        if S < best_S_global:
            best_S_global = S
            best_subset = subset
            best_center_global = center
            best_beta_global = beta

    if best_subset is None:
        # Fallback
        best_subset = list(CELLS_19[:N])
        pts = axial_to_cartesian(best_subset)
        best_S_global, best_center_global, best_beta_global = calc_S_for_subset(pts)

    pts = axial_to_cartesian(list(best_subset))
    cx, cy = best_center_global
    angle_deg = math.degrees(best_beta_global)
    
    # Safety margin
    S_final, _ = calc_S_for_subset(pts)[0] # Just use the calculated S
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(best_S_global)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
