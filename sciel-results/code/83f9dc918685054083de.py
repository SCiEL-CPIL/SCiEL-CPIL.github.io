import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute unit hexagon vertices (side 1, circumradius 1)
    # Angles 0, 60, 120, 180, 240, 300
    angles = np.arange(6) * (math.pi / 3.0)
    unit_hex_verts = np.array([
        [math.cos(a), math.sin(a)] for a in angles
    ])

    # Support function of unit hexagon in direction n (unit vector)
    # h(n) = max(v . n) for v in vertices
    # Since hexagon is symmetric, we can compute this efficiently.
    # For a regular hexagon, the support function is piecewise linear.
    # But for 6 fixed normals, we can precompute.
    
    def get_support(n):
        # n is a unit vector (2,)
        # Return max(v . n)
        projs = unit_hex_verts @ n
        return np.max(projs)

    def calc_S_for_config(pts, cx, cy, beta):
        """
        Calculate the minimum side length S of the outer hexagon
        centered at (cx, cy) with rotation beta (radians)
        that contains all inner hexagons at pts (N, 2) with rotation 0.
        """
        # Normals of the outer hexagon sides
        # Side normals are at angles beta + 30 + k*60
        normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        
        # For each normal n_k, we need:
        # max_i ( (c_i + v) . n_k ) <= c_outer . n_k + D
        # where D is the apothem of the outer hexagon.
        # c_outer = (cx, cy)
        
        c_outer = np.array([cx, cy])
        
        # Project centers of inner hexagons onto normals
        c_projs = pts @ normals.T # (N, 6)
        
        # For each normal, compute the support of the inner hexagon in that direction
        # Since inner hexagons have rotation 0, their support in direction n_k is fixed
        supports = np.array([get_support(n) for n in normals]) # (6,)
        
        # Max projection of each inner hexagon in direction k
        # max_i (c_i . n_k + support_k)
        max_projs = np.max(c_projs + supports[np.newaxis, :], axis=0) # (6,)
        
        # Similarly for min projection (or just use abs distance from center)
        # The condition is | (c_i + v) . n_k - c_outer . n_k | <= D
        # So we need D >= max_i,k | (c_i . n_k + support_k) - c_outer . n_k |
        # AND D >= max_i,k | (c_i . n_k - support_k) - c_outer . n_k | ?
        # No, the support function gives the max extent in direction n.
        # The min extent is -support(-n) = -support(n) due to symmetry.
        # So the range of projections for inner hexagon i in direction n_k is [c_i . n_k - s_k, c_i . n_k + s_k]
        # where s_k = support(n_k).
        
        # So we need:
        # max_i (c_i . n_k + s_k) <= c_outer . n_k + D
        # min_i (c_i . n_k - s_k) >= c_outer . n_k - D
        
        c_outer_projs = c_outer @ normals.T # (6,)
        
        D_req_max = np.max(max_projs - c_outer_projs)
        
        min_projs = np.min(c_projs - supports[np.newaxis, :], axis=0)
        D_req_min = np.max(c_outer_projs - min_projs)
        
        D = max(D_req_max, D_req_min)
        
        # S = (D + apothem_inner) / apothem_inner
        # Wait, the outer hexagon apothem is D.
        # The inner hexagons must fit.
        # The "width" of the inner hexagon in the normal direction is 2 * apothem_inner? No.
        # The support function s_k is the distance from center to the farthest vertex in direction n_k.
        # For a unit hexagon, s_k depends on the angle of n_k relative to the hexagon.
        # The apothem of the inner hexagon is APOTHEM.
        # The outer hexagon apothem must be at least the distance from outer center to the farthest point of any inner hexagon.
        # So D is the required apothem of the outer hexagon.
        # Side length S = 2 * D / sqrt(3) ?
        # Apothem a = S * sqrt(3) / 2 => S = 2 * a / sqrt(3)
        
        S = 2.0 * D / math.sqrt(3.0)
        return S

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e100
            
        pts = params[:22].reshape(11, 2)
        cx = params[22]
        cy = params[23]
        beta = params[24]
        
        # 1. Disjointness penalty
        # Use a smooth approximation of the collision detection.
        # For two regular hexagons of side 1, the minimum distance between centers
        # for them to be disjoint is not constant.
        # However, if we assume they are aligned (rotation 0), the minimum distance is 2 * apothem = sqrt(3) ~ 1.732.
        # If we allow rotation, it's complex.
        # Let's use a conservative penalty based on the distance between centers.
        # If dist < 2 * apothem, they might overlap.
        # A better proxy: distance between centers >= 2 * (sqrt(3)/2) * (1 + epsilon)?
        # Actually, the minimum distance between centers of two touching aligned hexagons is sqrt(3).
        # If we use a penalty for dist < sqrt(3), we ensure they are disjoint in the aligned case.
        # But if they rotate, they might need more space.
        # Given the time constraint, let's assume the optimal packing keeps them aligned or close to it.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(11, dtype=bool)
        min_dist_required = math.sqrt(3.0) * 1.01 # Slight margin
        overlaps = np.maximum(0, min_dist_required - dist_matrix)
        overlap_penalty = np.sum(overlaps[mask] ** 2)
        
        # 2. Calculate S
        S = calc_S_for_config(pts, cx, cy, beta)
        
        # 3. Total cost
        cost = S + 100.0 * overlap_penalty
        return cost

    # Generate diverse initial configurations
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    base_configs = [
        # "Flower" + 5
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)],
        # "Triangle" 3x3 + 2
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)],
        # "Line" variations
        [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)],
        # "Compact"
        [(0,0), (1,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)],
        # "Diagonal"
        [(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)],
        # "Star"
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)],
        # "Cross"
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (1,2), (-1,2), (0,3)],
        # "Block"
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (1,3)],
        # "L-Shape"
        [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2), (2,2), (0,3), (1,3), (0,4), (1,4)],
        # "T-Shape"
        [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3), (3,3)],
    ]

    initial_sets = [axial_to_cartesian(c) for c in base_configs]
    
    # Add jittered versions
    random_sets = []
    for pts in initial_sets[:5]:
        jittered = pts + np.random.normal(0, 0.1, (N, 2))
        random_sets.append(jittered)
        
    all_initial_sets = initial_sets + random_sets

    best_S = float('inf')
    best_params = None

    for i, pts_init in enumerate(all_initial_sets):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        cx_init = np.mean(pts_init[:, 0])
        cy_init = np.mean(pts_init[:, 1])
        beta_init = 0.0
        
        x0 = np.concatenate([pts_init.flatten(), [cx_init, cy_init, beta_init]])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_S:
                p = res.x
                pts_opt = p[:22].reshape(11, 2)
                cx_opt = p[22]
                cy_opt = p[23]
                beta_opt = p[24]
                
                # Verify validity
                diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
                dist_matrix = np.linalg.norm(diff, axis=2)
                mask = ~np.eye(11, dtype=bool)
                min_dist = np.min(dist_matrix[mask])
                
                if min_dist >= math.sqrt(3.0) * 0.99: # Tolerance
                    S_true = calc_S_for_config(pts_opt, cx_opt, cy_opt, beta_opt)
                    if S_true < best_S:
                        best_S = S_true
                        best_params = (pts_opt, cx_opt, cy_opt, beta_opt)
                        
        except Exception:
            continue

    if best_params is None:
        # Fallback to best initial config
        for pts_init in all_initial_sets:
            cx_init = np.mean(pts_init[:, 0])
            cy_init = np.mean(pts_init[:, 1])
            beta_init = 0.0
            
            # Check validity
            diff = pts_init[:, np.newaxis, :] - pts_init[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(11, dtype=bool)
            min_dist = np.min(dist_matrix[mask])
            
            if min_dist >= math.sqrt(3.0) * 0.99:
                S_true = calc_S_for_config(pts_init, cx_init, cy_init, beta_init)
                if S_true < best_S:
                    best_S = S_true
                    best_params = (pts_init, cx_init, cy_init, beta_init)

    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    pts_opt, cx_opt, cy_opt, beta_opt = best_params
    
    # Add tiny epsilon to S to avoid boundary issues
    final_S = best_S * (1.0 + 1e-9)
    
    inner = []
    for x, y in pts_opt:
        inner.append([float(x), float(y), float(0.0)])
        
    center = [float(cx_opt), float(cy_opt)]
    side = float(final_S)
    rotation = float(math.degrees(beta_opt))
    
    return inner, center, side, rotation
