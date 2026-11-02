import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 3.0
EPS = 1e-9

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

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3), (1, 3), (-1, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 2), (3, 2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 2), (-1, 2), (1, -1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_32)
    
    # New candidates to ensure distinctiveness
    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_33)
    
    axial_34 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4)
    ]
    candidates.append(axial_34)
    
    axial_35 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    candidates.append(axial_35)
    
    axial_36 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_36)
    
    # Additional diverse candidates
    axial_37 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_37)
    
    axial_38 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_38)
    
    axial_39 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1), (0, 2)
    ]
    candidates.append(axial_39)
    
    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_40)

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
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def calc_S_fast(pts, beta):
        """Fast exact computation of minimum enclosing hexagon for given beta.
        Uses the fact that optimal center minimizes max projection, which can be
        solved by checking all pairs of constraints."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        # Quick approach: use the centroid as initial guess, then iterate
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        
        # Compute D for this center
        projs = (pts - np.array([cx0, cy0])) @ normals.T
        D0 = float(np.max(np.abs(projs)))
        
        # Try a few nearby centers
        best_D = D0
        best_c = (cx0, cy0)
        
        # Grid search around centroid
        step = 0.1
        for dx in [-2*step, -step, 0, step, 2*step]:
            for dy in [-2*step, -step, 0, step, 2*step]:
                cx_try = cx0 + dx
                cy_try = cy0 + dy
                projs_try = (pts - np.array([cx_try, cy_try])) @ normals.T
                D_try = float(np.max(np.abs(projs_try)))
                if D_try < best_D:
                    best_D = D_try
                    best_c = (cx_try, cy_try)
        
        # Refine with local optimization
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj(c):
                projs = (pts - np.array(c)) @ normals.T
                return float(np.max(np.abs(projs)))
            
            try:
                res = minimize(obj, best_c, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_D:
                    best_D = res.fun
                    best_c = (res.x[0], res.x[1])
            except Exception:
                pass
        
        S = (best_D + APOTHEM) / APOTHEM
        return S, best_c

    def local_search_refine(pts, max_iters=20000):
        """
        Refines the configuration using local search.
        Assumes all inner hexagons are aligned with the outer hexagon (rotation 0).
        Optimizes positions and outer center.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return pts, (0,0), float('inf')
        
        # Normals for the hexagon (rotation 0)
        normals = np.array([
            (math.cos(math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        n_pts = len(pts)
        current_pts = pts.copy()
        current_center = np.array([np.mean(pts[:, 0]), np.mean(pts[:, 1])])
        
        def get_S(pts, center):
            # S = 1 + 2/√3 * max_proj
            # max_proj = max_k max_i | (p_i - c) . n_k |
            rel = pts - center
            projs = np.abs(rel @ normals.T)
            max_proj = np.max(projs)
            return 1.0 + (2.0 / math.sqrt(3.0)) * max_proj, max_proj

        def check_overlap(pts, center):
            # Overlap if dist < sqrt(3) in any of the 6 directions?
            # No, overlap if (p_i - p_j) is in the collision hexagon.
            # Collision hexagon is defined by |v . n_k| <= sqrt(3) for all k.
            # So NO overlap if there EXISTS k such that |(p_i - p_j) . n_k| > sqrt(3).
            # We want to ensure NO overlap, so we must ensure that for all i!=j,
            # there is at least one k such that |(p_i - p_j) . n_k| > sqrt(3).
            # Wait, the condition for NON-overlap is that the relative vector is OUTSIDE the collision shape.
            # The collision shape is the intersection of the 6 strips |v.n_k| <= sqrt(3).
            # So v is inside if |v.n_k| <= sqrt(3) for ALL k.
            # So v is outside if there EXISTS k such that |v.n_k| > sqrt(3).
            
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    v = pts[i] - pts[j]
                    projs = np.abs(v @ normals.T)
                    # If max projection is <= sqrt(3), they overlap (or touch)
                    if np.max(projs) <= math.sqrt(3.0) + 1e-9:
                        return False
            return True

        current_S, _ = get_S(current_pts, current_center)
        best_S = current_S
        best_pts = current_pts.copy()
        best_center = current_center.copy()
        
        T = 0.1
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Random move
            if np.random.rand() < 0.1:
                # Move center
                dx = np.random.normal(0, 0.1)
                dy = np.random.normal(0, 0.1)
                new_center = current_center + np.array([dx, dy])
                new_pts = current_pts
            else:
                # Move point
                i = np.random.randint(n_pts)
                dx = np.random.normal(0, 0.1)
                dy = np.random.normal(0, 0.1)
                new_pts = current_pts.copy()
                new_pts[i] = current_pts[i] + np.array([dx, dy])
                new_center = current_center
            
            if not check_overlap(new_pts, new_center):
                T *= 0.999
                continue
                
            new_S, _ = get_S(new_pts, new_center)
            
            if new_S < current_S or np.random.rand() < math.exp(-(new_S - current_S) / T):
                current_S = new_S
                current_pts = new_pts
                current_center = new_center
                
                if current_S < best_S:
                    best_S = current_S
                    best_pts = current_pts.copy()
                    best_center = current_center.copy()
            
            T *= 0.999
            if T < 1e-5:
                T = 1e-5
                
        return best_pts, best_center, best_S

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # First pass: quick evaluation using fast method
    quick_results = []
    for i, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        best_quick_S = float('inf')
        best_quick_beta = 0.0
        best_quick_center = None
        for j in range(13):
            beta = j * (math.pi / 6) / 12
            S, center = calc_S_fast(pts, beta)
            if S < best_quick_S:
                best_quick_S = S
                best_quick_beta = beta
                best_quick_center = center
        quick_results.append((best_quick_S, i, pts, best_quick_beta, best_quick_center))
    
    quick_results.sort(key=lambda x: x[0])
    
    # Use local search on top candidates
    max_detailed = min(len(quick_results), 10)
    for idx in range(max_detailed):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        quick_S, axial_idx, pts, quick_beta, quick_center = quick_results[idx]
        
        # Run local search assuming aligned hexagons (beta=0 effectively for the shape)
        # The local search optimizes positions and center for beta=0.
        # We can also try to rotate the whole result later?
        # For now, assume beta=0 is good enough for the local search to find a tight packing.
        refined_pts, refined_center, refined_S = local_search_refine(pts)
        
        if refined_S < best_global_S:
            best_global_S = refined_S
            best_global_pts = refined_pts
            best_global_beta = 0.0 # Aligned
            best_global_center = refined_center

    # Fallback if local search failed or didn't improve
    if best_global_center is None or best_global_S == float('inf'):
        # Use the best from quick pass
        if quick_results:
            best_quick_S, _, best_global_pts, best_global_beta, best_global_center = quick_results[0]
            best_global_S = best_quick_S
        else:
            best_global_pts = axial_to_cartesian(candidates[0])
            best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
            best_global_beta = 0.0
            max_proj = 0
            for k in range(6):
                ang = math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Verify and scale slightly
    pts_scaled = best_global_pts
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
