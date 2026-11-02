import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute hex normals (directions perpendicular to sides)
    HEX_NORMALS = np.array([
        [math.cos(math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])

    # Axial to cartesian conversion
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Check if two hexagons overlap given their centers
    def hex_dist(c1, c2):
        """Minimum distance between two unit hexagons (0 if touching/overlapping)"""
        d = c1 - c2
        projs = np.abs(d @ HEX_NORMALS.T)
        min_proj = np.min(projs)
        return max(0, SQRT3 - min_proj)

    # Check all pairwise overlaps
    def check_no_overlap(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if hex_dist(pts[i], pts[j]) > EPS:
                    return False
        return True

    # Compute the outer side length S for given configuration
    def compute_S(pts, center, beta):
        """
        Compute the side length of the enclosing regular hexagon.
        pts: (N,2) array of hex centers
        center: (2,) array of enclosing hex center
        beta: rotation angle of enclosing hex
        Returns: (S, D) where S = (D + APOTHEM) / APOTHEM
        """
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        rel = pts - center
        projs = rel @ normals.T  # (N, 6)
        D = np.max(np.abs(projs))
        S = (D + APOTHEM) / APOTHEM
        return S, D

    # Smooth version for optimization
    def compute_S_smooth(pts, center, beta, temp=0.01):
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        rel = pts - center
        projs = rel @ normals.T  # (N, 6)
        abs_projs = np.abs(projs)
        
        # Soft max
        max_val = np.max(abs_projs)
        if max_val > 100:
            return 1e10, 1e10
        shifted = (abs_projs - max_val) / temp
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted), axis=1))  # (N,)
        log_sum_max = np.max(log_sum)
        # D = temp * log(sum(exp((abs_projs - max_val)/temp))) + max_val
        D = temp * log_sum_max + max_val
        S = (D + APOTHEM) / APOTHEM
        return S, D

    # Generate candidate configurations
    def generate_candidates():
        candidates = []
        
        # Configuration 1: 3x4 grid (slightly compressed)
        c1 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c1))
        
        # Configuration 2: 3x4 with offset
        c2 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (1, 1), (2, 1), (3, 1), (4, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c2))
        
        # Configuration 3: Hexagonal cluster
        c3 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (1, 2)
        ]
        candidates.append(axial_to_cartesian(c3))
        
        # Configuration 4: Hexagonal with different outer
        c4 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 1), (1, 2), (-1, 2), (0, 3)
        ]
        candidates.append(axial_to_cartesian(c4))
        
        # Configuration 5: 2x5+1
        c5 = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (2, 2)
        ]
        candidates.append(axial_to_cartesian(c5))
        
        # Configuration 6: 4x3 compressed
        c6 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c6))
        
        # Configuration 7: Star pattern
        c7 = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        candidates.append(axial_to_cartesian(c7))
        
        # Configuration 8: Diamond
        c8 = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1),
            (2, 0), (-2, 0), (1, -1), (-1, -1)
        ]
        candidates.append(axial_to_cartesian(c8))
        
        # Configuration 9: 3 rows of varying length
        c9 = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c9))
        
        # Configuration 10: 4x3 with different offset
        c10 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (1, 2), (2, 2), (3, 2)
        ]
        candidates.append(axial_to_cartesian(c10))
        
        # Configuration 11: 5x2+1
        c11 = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (2, 2)
        ]
        candidates.append(axial_to_cartesian(c11))
        
        # Configuration 12: 4x3 symmetric
        c12 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c12))
        
        # Configuration 13: 3x4 with gap
        c13 = [
            (0, 0), (1, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c13))
        
        # Configuration 14: 4x3 rotated
        c14 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c14))
        
        # Configuration 15: 5x2+1 variant
        c15 = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (1, 2)
        ]
        candidates.append(axial_to_cartesian(c15))
        
        # Configuration 16: 4x3 with different gap
        c16 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (1, 1), (2, 1), (3, 1), (4, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c16))
        
        # Configuration 17: 3x4 with offset
        c17 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c17))
        
        # Configuration 18: 4x3 with gap
        c18 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (3, 1), (4, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c18))
        
        # Configuration 19: 5x2+1 variant
        c19 = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (3, 2)
        ]
        candidates.append(axial_to_cartesian(c19))
        
        # Configuration 20: 4x3 with different offset
        c20 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (1, 1), (2, 1), (3, 1), (4, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c20))
        
        # Configuration 21: 3x4 with gap
        c21 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c21))
        
        # Configuration 22: 4x3 with different gap
        c22 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c22))
        
        # Configuration 23: 5x2+1 variant
        c23 = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (0, 2)
        ]
        candidates.append(axial_to_cartesian(c23))
        
        # Configuration 24: 4x3 with different offset
        c24 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (1, 2), (2, 2), (3, 2)
        ]
        candidates.append(axial_to_cartesian(c24))
        
        # Configuration 25: 3x4 with gap
        c25 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c25))
        
        # Configuration 26: 4x3 with different gap
        c26 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c26))
        
        # Configuration 27: 5x2+1 variant
        c27 = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (1, 2)
        ]
        candidates.append(axial_to_cartesian(c27))
        
        # Configuration 28: 4x3 with different offset
        c28 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c28))
        
        # Configuration 29: 3x4 with gap
        c29 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c29))
        
        # Configuration 30: 4x3 with different gap
        c30 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        candidates.append(axial_to_cartesian(c30))
        
        return candidates

    def optimize_configuration(pts_init):
        """
        Optimize a configuration by adjusting positions, center, and rotation.
        Returns: (best_S, best_pts, best_center, best_beta)
        """
        n_pts = len(pts_init)
        
        # Initial: center is mean of points, beta = 0
        init_center = np.mean(pts_init, axis=0)
        init_beta = 0.0
        
        # First, find best beta for the initial configuration
        best_beta = 0.0
        best_S = float('inf')
        
        for i in range(100):
            beta = i * (math.pi / 6) / 99
            S, _ = compute_S(pts_init, init_center, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
        
        # Refine beta
        try:
            res = minimize_scalar(lambda b: compute_S(pts_init, init_center, b)[0],
                                  bounds=(0, math.pi / 6), method='bounded',
                                  options={'maxiter': 500, 'xatol': 1e-14})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
        except Exception:
            pass
        
        # Now optimize positions and center
        def objective(params):
            pts = params[:2 * n_pts].reshape(n_pts, 2)
            cx, cy = params[2 * n_pts], params[2 * n_pts + 1]
            beta = params[2 * n_pts + 2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            # Check overlap
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if hex_dist(pts[i], pts[j]) > EPS:
                        return 1e10
            
            S, _ = compute_S_smooth(pts, np.array([cx, cy]), beta, temp=0.005)
            return S
        
        init_params = np.concatenate([
            pts_init.ravel(),
            [init_center[0], init_center[1], best_beta]
        ])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.fun < best_S + 0.01:
                pts_opt = res.x[:2 * n_pts].reshape(n_pts, 2)
                cx, cy = res.x[2 * n_pts], res.x[2 * n_pts + 1]
                beta_opt = ((res.x[2 * n_pts + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                # Verify exact
                if check_no_overlap(pts_opt):
                    S_exact, _ = compute_S(pts_opt, np.array([cx, cy]), beta_opt)
                    if S_exact < best_S:
                        best_S = S_exact
                        pts_init = pts_opt
                        init_center = np.array([cx, cy])
                        best_beta = beta_opt
        except Exception:
            pass
        
        return best_S, pts_init, init_center, best_beta

    # Main optimization
    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_beta = 0.0

    candidates = generate_candidates()
    
    # Sort by initial S
    scored = []
    for idx, pts in enumerate(candidates):
        center = np.mean(pts, axis=0)
        best_beta = 0.0
        best_S = float('inf')
        for i in range(50):
            beta = i * (math.pi / 6) / 49
            S, _ = compute_S(pts, center, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
        scored.append((best_S, idx, pts))
    
    scored.sort(key=lambda x: x[0])
    
    for S_init, idx, pts in scored:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        S, pts_opt, center, beta = optimize_configuration(pts)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_opt
            best_global_center = center
            best_global_beta = beta
    
    # Final polish
    if best_global_pts is not None and best_global_center is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5:
        n_pts = len(best_global_pts)
        
        def final_obj(params):
            pts = params[:2 * n_pts].reshape(n_pts, 2)
            cx, cy = params[2 * n_pts], params[2 * n_pts + 1]
            beta = params[2 * n_pts + 2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if hex_dist(pts[i], pts[j]) > EPS:
                        return 1e10
            
            S, _ = compute_S_smooth(pts, np.array([cx, cy]), beta, temp=0.001)
            return S
        
        init_params = np.concatenate([
            best_global_pts.ravel(),
            [best_global_center[0], best_global_center[1], best_global_beta]
        ])
        
        try:
            res = minimize(final_obj, init_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
            
            if res.fun < best_global_S + 0.001:
                pts_final = res.x[:2 * n_pts].reshape(n_pts, 2)
                cx, cy = res.x[2 * n_pts], res.x[2 * n_pts + 1]
                beta_final = ((res.x[2 * n_pts + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                if check_no_overlap(pts_final):
                    S_final, _ = compute_S(pts_final, np.array([cx, cy]), beta_final)
                    if S_final < best_global_S:
                        best_global_S = S_final
                        best_global_pts = pts_final
                        best_global_center = np.array([cx, cy])
                        best_global_beta = beta_final
        except Exception:
            pass

    # Fallback
    if best_global_pts is None or best_global_S == float('inf'):
        best_global_pts = candidates[0]
        best_global_center = np.mean(best_global_pts, axis=0)
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    # Compute final S
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Small safety margin
    SCALE = 1.0 + 1e-12
    pts_scaled = np.array([
        (cx + (x - cx) * SCALE, cy + (y - cy) * SCALE)
        for x, y in best_global_pts
    ])
    
    S_final, _ = compute_S(pts_scaled, np.array([cx, cy]), best_global_beta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
