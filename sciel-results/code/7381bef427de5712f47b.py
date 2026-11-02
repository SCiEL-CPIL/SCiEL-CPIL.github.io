import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Hexagonal grid neighbors for side length 1
    # Axial coordinates: (q, r) -> cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    NEIGHBORS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def compute_S(pts, beta, cx, cy):
        """Compute the required outer hexagon side length for given configuration."""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            for px, py in pts:
                proj = abs((px - cx) * nx + (py - cy) * ny)
                if proj > max_proj:
                    max_proj = proj
        
        S = (max_proj + APOTHEM) / APOTHEM
        return S, max_proj

    def find_optimal_center_and_beta(pts, init_cx=None, init_cy=None, init_beta=None):
        """Find optimal center and rotation for a given set of points."""
        if init_cx is None:
            init_cx = float(np.mean(pts[:, 0]))
            init_cy = float(np.mean(pts[:, 1]))
        if init_beta is None:
            init_beta = 0.0
        
        def objective(params):
            cx, cy, beta = params
            S, _ = compute_S(pts, beta, cx, cy)
            return S
        
        best_S = float('inf')
        best_params = [init_cx, init_cy, init_beta]
        
        # Try multiple starting points
        start_betas = [init_beta, 0.0, math.pi/12, math.pi/6, math.pi/18, -math.pi/12]
        for sb in start_betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            init = [init_cx, init_cy, sb]
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        
        # Polish with L-BFGS-B
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                res = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 500, 'gtol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        
        return best_S, best_params

    # Generate candidate configurations using hexagonal grid packing
    # Strategy: place hexagons on a hex grid, trying different cluster shapes
    candidates = []
    
    # Configuration 1: Central hex + 6 neighbors + 4 more
    # This is a hexagonal cluster of 11
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (2, -1)
    ]
    candidates.append(axial_1)

    # Configuration 2: More elongated
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2)
    ]
    candidates.append(axial_2)

    # Configuration 3: Compact triangular
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_3)

    # Configuration 4: Diamond shape
    axial_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_4)

    # Configuration 5: Asymmetric cluster
    axial_5 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_5)

    # Configuration 6: Star pattern
    axial_6 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-1, 2), (-2, 0), (1, -2)
    ]
    candidates.append(axial_6)

    # Configuration 7: Wide bar
    axial_7 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (2, 2)
    ]
    candidates.append(axial_7)

    # Configuration 8: T-shape
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (1, -2)
    ]
    candidates.append(axial_8)

    # Configuration 9: Compact square-like
    axial_9 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_9)

    # Configuration 10: Zigzag
    axial_10 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (2, 3), (3, 3), (4, 3)
    ]
    candidates.append(axial_10)

    # Configuration 11: Very compact
    axial_11 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (1, 3)
    ]
    candidates.append(axial_11)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for i, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        
        # Quick estimate: bounding box in hex direction
        cx_est = float(np.mean(pts[:, 0]))
        cy_est = float(np.mean(pts[:, 1]))
        
        S, params = find_optimal_center_and_beta(pts, init_cx=cx_est, init_cy=cy_est, init_beta=0.0)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_center = (params[0], params[1])
            best_global_beta = params[2]
    
    if best_global_center is None or best_global_S == float('inf'):
        # Fallback: simple hexagonal cluster
        pts = axial_to_cartesian(candidates[0])
        best_global_pts = pts
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = compute_S(pts, 0.0, best_global_center[0], best_global_center[1])[0]

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Compute final S
    S_final, _ = compute_S(best_global_pts, best_global_beta, cx, cy)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
