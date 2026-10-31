import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    SCALE = 1.001
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    rel_pts = pts - centroid

    # Smooth objective: use log-sum-exp to smooth the max over 6 normals
    def smooth_max_S(beta, alpha=50.0):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)
        # projections shape (6, 11)
        projections = rel_pts @ normals.T
        abs_proj = np.abs(projections)
        # For each normal, required S = (max_abs_proj + APOTHEM) / APOTHEM
        max_abs = np.max(abs_proj, axis=1)  # shape (6,)
        req_S = (max_abs + APOTHEM) / APOTHEM
        # Smooth max via log-sum-exp
        m = np.max(req_S)
        if m > 1e10:
            return float('inf')
        smooth = m + math.log(np.sum(np.exp(req_S - m))) / alpha
        return smooth

    # Exact objective for final evaluation
    def exact_S(beta):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)
        projections = rel_pts @ normals.T
        abs_proj = np.abs(projections)
        max_abs = np.max(abs_proj, axis=1)
        req_S = (max_abs + APOTHEM) / APOTHEM
        return np.max(req_S)

    # Multi-start for outer rotation
    best_beta = 0.0
    best_S = exact_S(0.0)
    
    # Try multiple initial angles
    init_angles = [0.0, math.pi/12, math.pi/6, math.pi/8, math.pi/10, 
                   math.pi/18, math.pi/24, math.pi/15, math.pi/16, math.pi/20]
    
    for init in init_angles:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        try:
            res = minimize(smooth_max_S, [init], method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < float('inf'):
                # Refine with exact objective using bounded scalar minimization
                b = res.x[0]
                b = b % (math.pi / 3)  # Period is pi/3
                b = max(0, min(b, math.pi / 6))
                res2 = minimize_scalar(exact_S, bounds=(max(0, b - 0.05), min(math.pi/6, b + 0.05)),
                                       method='bounded', options={'maxiter': 200, 'xatol': 1e-10})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_beta = res2.x
        except Exception:
            pass

    # Joint refinement: allow small shifts of inner hexagons to reduce S
    # Parameters: 6D shift vector (dx, dy) for each of 6 outer hexagons (skip center)
    # Actually, let's allow small radial shifts for all 11
    # Use a simpler approach: try small perturbations of the lattice scale
    
    # Alternative: optimize the lattice scale factor
    def S_with_scale(scale, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        # Recompute points with new scale
        new_pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0) * scale
            y = 1.5 * r * scale
            new_pts.append((x, y))
        new_pts = np.array(new_pts, dtype=np.float64)
        new_centroid = np.mean(new_pts, axis=0)
        new_rel = new_pts - new_centroid
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)
        projections = new_rel @ normals.T
        abs_proj = np.abs(projections)
        max_abs = np.max(abs_proj, axis=1)
        req_S = (max_abs + APOTHEM) / APOTHEM
        return np.max(req_S)

    # Optimize scale factor around 1.001
    best_scale = SCALE
    best_S_final = best_S
    
    for scale_init in [0.999, 1.0, 1.001, 1.002, 1.003]:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        try:
            def obj_scale(s):
                return S_with_scale(s, best_beta)
            res_s = minimize_scalar(obj_scale, bounds=(0.995, 1.005), method='bounded',
                                   options={'maxiter': 100, 'xatol': 1e-10})
            if res_s.fun < best_S_final:
                best_S_final = res_s.fun
                best_scale = res_s.x
        except Exception:
            pass

    # Final joint optimization: optimize both beta and scale together
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            def obj_joint(params):
                beta, scale = params
                return S_with_scale(scale, beta)
            
            init_params = [best_beta, best_scale]
            res_joint = minimize(obj_joint, init_params, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
            if res_joint.fun < best_S_final:
                best_S_final = res_joint.fun
                best_beta = res_joint.x[0]
                best_scale = res_joint.x[1]
        except Exception:
            pass

    # Reconstruct final points with optimal scale
    final_pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * best_scale
        y = 1.5 * r * best_scale
        final_pts.append((x, y))
    final_pts = np.array(final_pts, dtype=np.float64)
    final_centroid = np.mean(final_pts, axis=0)
    
    # Final exact evaluation
    final_S = S_with_scale(best_scale, best_beta)
    
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in final_pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(final_centroid[0]), float(final_centroid[1])]
    side = float(final_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
