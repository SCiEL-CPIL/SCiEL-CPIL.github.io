import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Corner-filling template: 11 unit hexagons at the 6 vertices and
    # 3 edge-midpoints of a unit-hexagon ring (axial coords).
    # This is a continuous, space-filling layout that explicitly uses
    # the boundary of the outer hexagon.
    axial_template = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (1, 1), (-1, -1)
    ]
    
    # Convert axial to cartesian for the template (side=1 hexes)
    pts = np.array([
        (math.sqrt(3) * (q + r / 2.0), 1.5 * r)
        for q, r in axial_template
    ])
    
    # Center of template
    cx0, cy0 = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    pts_centered = pts - np.array([cx0, cy0])
    
    def exact_S(beta, cx, cy):
        """Exact outer side length for given rotation and center."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        projs = (pts_centered + np.array([cx0 - cx, cy0 - cy])) @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_S(beta, cx, cy, temp=0.05):
        """Smooth surrogate for optimization."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        projs = (pts_centered + np.array([cx0 - cx, cy0 - cy])) @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        shifted = np.clip((all_projs - max_val) / temp, -50, 50)
        D = temp * (np.log(np.sum(np.exp(shifted))) + max_val / temp)
        return (D + APOTHEM) / APOTHEM

    # Optimize over beta (rotation) and center (cx, cy)
    best_S = float('inf')
    best_beta = 0.0
    best_cx = cx0
    best_cy = cy0

    # Coarse grid over beta
    for i in range(121):
        beta = i * (math.pi / 6) / 120
        S = exact_S(beta, 0.0, 0.0)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_cx = 0.0
            best_cy = 0.0

    # Refine beta with scalar minimization (center fixed at 0 for now)
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(
                lambda b: exact_S(b, 0.0, 0.0),
                bounds=(0, math.pi / 6),
                method='bounded',
                options={'maxiter': 2000, 'xatol': 1e-15}
            )
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                best_cx = 0.0
                best_cy = 0.0
    except Exception:
        pass

    # Optimize center for fixed beta using smooth surrogate + Nelder-Mead
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def center_obj(params):
                cx, cy = params
                return smooth_S(best_beta, cx, cy, temp=0.05)
            
            res = minimize_scalar(
                lambda p: center_obj([p, 0.0]),
                bounds=(-1, 1),
                method='bounded',
                options={'maxiter': 1000, 'xatol': 1e-15}
            )
            if res.fun < 1e10:
                best_cx = res.x
                # Refine cy
                res2 = minimize_scalar(
                    lambda p: center_obj([best_cx, p]),
                    bounds=(-1, 1),
                    method='bounded',
                    options={'maxiter': 1000, 'xatol': 1e-15}
                )
                if res2.fun < 1e10:
                    best_cy = res2.x
                    S_exact = exact_S(best_beta, best_cx, best_cy)
                    if S_exact < best_S:
                        best_S = S_exact
    except Exception:
        pass

    # Final polish: joint (beta, cx, cy) with smooth surrogate
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            from scipy.optimize import minimize
            
            def joint_obj(params):
                beta, cx, cy = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return smooth_S(beta, cx, cy, temp=0.02)
            
            starts = [
                [best_beta, best_cx, best_cy],
                [best_beta + 0.01, best_cx, best_cy],
                [best_beta - 0.01, best_cx, best_cy],
                [best_beta, best_cx + 0.01, best_cy],
                [best_beta, best_cx - 0.01, best_cy],
                [best_beta, best_cx, best_cy + 0.01],
                [best_beta, best_cx, best_cy - 0.01],
                [0.0, 0.0, 0.0],
                [math.pi/12, 0.0, 0.0],
                [math.pi/6, 0.0, 0.0],
            ]
            
            for init in starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                res = minimize(joint_obj, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < 1e10:
                    beta_r, cx_r, cy_r = res.x
                    beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    S_exact = exact_S(beta_r, cx_r, cy_r)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_beta = beta_r
                        best_cx = cx_r
                        best_cy = cy_r
    except Exception:
        pass

    # Final exact polish on beta only with tight tolerance
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(
                lambda b: exact_S(b, best_cx, best_cy),
                bounds=(0, math.pi / 6),
                method='bounded',
                options={'maxiter': 5000, 'xatol': 1e-16}
            )
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
    except Exception:
        pass

    angle_deg = math.degrees(best_beta)
    
    # Construct inner hexagons at the template positions
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
