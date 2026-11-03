import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)
    
    # ... (keep existing candidates) ...

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx0, cy0])

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        max_proj = 0.0
        for nx, ny in normals:
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))

        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S, (cx0, cy0), D

    def optimize_theta(pts):
        best_S = float('inf')
        best_beta = 0.0
        
        # Coarse grid
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, _, _ = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta

        # Refine
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
        except Exception:
            pass
        return best_S, best_beta

    # Evaluate all candidates
    scored_candidates = []
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta = optimize_theta(pts)
        scored_candidates.append((S, pts, beta))

    scored_candidates.sort(key=lambda x: x[0])
    
    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    # Take top 3 candidates and try to optimize their positions
    for S_init, pts_init, beta_init in scored_candidates[:3]:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Flatten points to 1D array for optimization
        x0 = pts_init.flatten()
        
        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return float('inf')
            pts = params.reshape(-1, 2)
            # Optimize theta for these points
            S_opt, _ = optimize_theta(pts)
            return S_opt

        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_pts = res.x.reshape(-1, 2)
                # Re-optimize theta for the final points
                best_S, best_beta = optimize_theta(best_pts)
                cx0 = np.mean(best_pts[:, 0])
                cy0 = np.mean(best_pts[:, 1])
                best_center = (cx0, cy0)
        except Exception:
            pass

    # If no optimization improved, use the best initial candidate
    if best_pts is None:
        best_S, best_pts, best_beta = scored_candidates[0]
        cx0 = np.mean(best_pts[:, 0])
        cy0 = np.mean(best_pts[:, 1])
        best_center = (cx0, cy0)

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    # Scale slightly to ensure strict disjointness if needed, though integer coords are safe
    # The problem states "disjoint", usually meaning non-overlapping interiors.
    # With integer axial coords, distance between centers is >= sqrt(3) (edge touching) or more.
    # Unit hexagons have width sqrt(3). So edge touching is valid.
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    
    # Recalculate side length precisely
    max_proj = 0
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (best_pts[:, 0] - cx) * nx + (best_pts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
    D_scaled = max_proj
    side = float((D_scaled + APOTHEM) / APOTHEM)
    rotation = float(angle_deg)

    return inner, center, side, rotation
