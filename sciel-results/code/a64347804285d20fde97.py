import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Core 7 hexagons
    core = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    
    # Candidate outer positions (12 neighbors of the ring)
    outer_candidates = [
        (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2), # Axial
        (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)  # Diagonal
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def rough_tightness(pts):
        """Estimate tightness based on max distance from centroid."""
        centroid = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        return np.max(dists)

    # Generate all combinations of 4 outer points
    all_combos = list(combinations(outer_candidates, 4))
    
    # Score and sort
    scored_combos = []
    for combo in all_combos:
        full_cluster = core + list(combo)
        pts = axial_to_cartesian(full_cluster)
        score = rough_tightness(pts)
        scored_combos.append((score, full_cluster))
        
    scored_combos.sort(key=lambda x: x[0])
    
    # Take top 20 unique configurations (check for duplicates? Combinations are unique)
    # But some might be symmetric. Let's just take top 20.
    top_candidates = [cluster for _, cluster in scored_combos[:20]]
    
    # Also include some hardcoded ones that might be good but not top in rough metric?
    # The rough metric is max dist from centroid. This is a good proxy for bounding box size.
    # Let's stick to the top 20.

    def ensure_no_overlap(pts, min_dist=math.sqrt(3) * 1.001):
        pts = pts.copy()
        for _ in range(100):
            moved = False
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_dist and d > 1e-12:
                        diff = (pts[i] - pts[j]) / d
                        pts[i] += diff * (min_dist - d) / 2.0
                        pts[j] -= diff * (min_dist - d) / 2.0
                        moved = True
                    elif d <= 1e-12:
                        pts[i] += [0.001, 0.001]
                        moved = True
            if not moved:
                break
        return pts

    def exact_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        projs = normals @ pts.T
        
        max_projs = projs.max(axis=1)
        min_projs = projs.min(axis=1)
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for k in range(6):
            A_ub.append([-normals[k,0], -normals[k,1], -1.0])
            b_ub.append(-max_projs[k])
            A_ub.append([normals[k,0], normals[k,1], -1.0])
            b_ub.append(min_projs[k])
        
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

    def exact_S_direct(pts, beta, cx, cy):
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        centered = pts - np.array([cx, cy])
        projs = normals @ centered.T
        max_proj = np.max(np.abs(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = exact_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi/6/72)
                hi = min(math.pi/6, best_beta + math.pi/6/72)
                res = minimize_scalar(lambda b: exact_S_for_beta(pts, b)[0], 
                                     bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = exact_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S_direct(pts, beta, cx, cy)
                
                centroid = np.mean(pts, axis=0)
                starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.03],
                    [best_center[0], best_center[1], best_beta - 0.03],
                    [centroid[0], centroid[1], 0.0],
                    [centroid[0], centroid[1], math.pi/12],
                ]
                
                for init in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in top_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        pts = ensure_no_overlap(pts)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to first candidate
        pts = axial_to_cartesian(top_candidates[0])
        pts = ensure_no_overlap(pts)
        best_global_pts = pts
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    pts_final = best_global_pts
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_final[:, 0] - cx) * nx + (pts_final[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
