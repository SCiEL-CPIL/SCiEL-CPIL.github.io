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

    # 1. Generate a diverse set of initial grid packings (axial coordinates)
    # These are known compact configurations for 11 hexagons.
    # Axial coords (q, r) -> Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
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

    # Convert to Cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    initial_sets = [axial_to_cartesian(c) for c in base_configs]
    
    # Add some random perturbations to the best grid to break symmetry
    random_sets = []
    for pts in initial_sets[:5]:
        # Small random jitter
        jittered = pts + np.random.normal(0, 0.1, (N, 2))
        random_sets.append(jittered)

    all_initial_sets = initial_sets + random_sets

    # Precompute unit hexagon vertices relative to center
    # Vertices at angles 0, 60, ... 300
    angles = np.arange(6) * (math.pi / 3.0)
    # Vertices of a unit hexagon (side 1) have radius 1
    unit_hex_verts = np.array([
        [math.cos(a), math.sin(a)] for a in angles
    ])

    # Function to check if two hexagons are disjoint
    # We use a conservative separation check.
    # For two regular hexagons of side 1, if the distance between centers is >= 2 * apothem * 2 = 2*sqrt(3) ~ 3.46, they are definitely disjoint?
    # Actually, the minimum distance between centers for disjoint hexagons depends on orientation.
    # If aligned, min distance is 2 * apothem = sqrt(3) ~ 1.732.
    # If rotated 30 deg, min distance is 2 * (sqrt(3)/2) * 2? No.
    # The "radius" of a hexagon is 1. The apothem is sqrt(3)/2.
    # If two hexagons touch, the distance between centers is at least the sum of their "radii" in the direction of the center line.
    # To be safe and fast, we can use a distance threshold.
    # However, for the optimization, we need a smooth penalty.
    
    # A better smooth penalty: 
    # Calculate the minimum distance between the boundaries.
    # Approximation: If dist < 2 * apothem, they might overlap.
    # Let's use a potential energy function that goes to infinity as they overlap.
    
    # For simplicity and speed in the inner loop of Nelder-Mead, we will use a simplified constraint:
    # The centers must be at least 2 * apothem * (1 + epsilon) apart? No, that's too restrictive for rotated hexes.
    # Actually, if we keep them on a grid-like structure, they are aligned.
    # If we allow rotation, it's complex.
    # Let's assume the inner hexagons stay aligned with the outer hexagon or each other?
    # The problem allows rotation.
    
    # Strategy: Optimize positions (x, y) for all 11 hexagons, assuming they all have the SAME rotation as the outer hexagon.
    # This is a strong assumption but often optimal for regular shapes.
    # If we allow individual rotations, the search space is 11*3 + 3 = 36D, which is too high for Nelder-Mead in 120s.
    # Let's stick to: All inner hexagons have the same rotation angle theta, and the outer hexagon has rotation theta.
    # Wait, the outer hexagon rotation and inner hexagon rotation don't have to be the same.
    # But usually, aligning them is optimal.
    # Let's optimize:
    # - 11 (x, y) pairs for centers
    # - 1 rotation angle for the outer hexagon (and inner ones)
    # Total 23 variables.
    
    # Actually, let's just optimize the 11 (x,y) positions and the outer hexagon center (cx, cy) and rotation beta.
    # Inner hexagons are fixed to rotation 0 relative to the global frame? No, let's fix inner rotation to 0 for now and see if we can beat 10.
    # If inner rotation is 0, the "effective" size of the cluster is fixed. We just need to find the best translation/rotation of the cluster.
    # But we can also move the inner hexagons relative to each other!
    
    # Let's define the objective function:
    # Minimize S (outer side length) subject to:
    # 1. All inner hexagons are inside the outer hexagon.
    # 2. Inner hexagons are disjoint.
    
    # We will use a penalty method.
    
    def objective(params):
        # params: x0, y0, ..., x10, y10, cx, cy, beta
        # 11*2 + 3 = 25 variables
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e100

        pts = params[:22].reshape(11, 2)
        cx = params[22]
        cy = params[23]
        beta = params[24]
        
        # 1. Penalty for inner hexagons being outside the outer hexagon
        # Outer hexagon: center (cx, cy), side S, rotation beta.
        # We want to find the minimum S such that all pts are inside.
        # Instead of calculating S directly, we can calculate the required S for each point and take the max.
        # But S is not in params. We can estimate S from the current configuration.
        # Or, we can fix S to a target and penalize violations?
        # Better: Calculate the minimal S required for the current pts and beta.
        # S = (max_proj + apothem) / apothem
        
        # Normals of the outer hexagon
        # The sides of the hexagon are perpendicular to vectors at angles beta + 30 + k*60
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)
        
        # Project all points onto normals
        # P_i . n_k
        projs = pts @ normals.T  # (11, 6)
        
        # For each normal k, the max projection of any point is max_j (P_j . n_k)
        # The min projection is min_j (P_j . n_k)
        # The width in direction k is max_j - min_j
        # The center of the hexagon must be such that the projections are within [-D, D]
        # Actually, the condition for a point P to be inside a hexagon centered at C with "radius" D (apothem of outer) is:
        # |(P - C) . n_k| <= D for all k
        # So, for all j, k: |(P_j - C) . n_k| <= D
        # This implies:
        # max_j,k |(P_j - C) . n_k| <= D
        
        # Let's compute the required D for the current C and pts
        rel_pts = pts - np.array([cx, cy])
        rel_projs = rel_pts @ normals.T
        abs_projs = np.abs(rel_projs)
        max_abs_proj = np.max(abs_projs)
        
        # Required D = max_abs_proj
        # S = (D + apothem_inner) / apothem_inner
        # Wait, the inner hexagons have side 1, so their apothem is sqrt(3)/2.
        # The outer hexagon must contain the entire inner hexagon.
        # The "radius" of the inner hexagon in direction n is apothem_inner + (projection of vertices)?
        # No, the support function of a regular hexagon of side 1 in direction n is:
        # h(n) = apothem_inner / cos(angle_diff) ? No.
        # The support function of a regular polygon is the distance from center to the tangent line.
        # For a regular hexagon, the support function is constant only if the direction is normal to a side.
        # Actually, the condition "inner hexagon is inside outer hexagon" is equivalent to:
        # For each of the 6 sides of the outer hexagon, the distance from the outer side to the farthest vertex of the inner hexagon must be >= 0.
        
        # This is complex to vectorize quickly for 11 hexagons.
        # Approximation: Treat inner hexagons as circles of radius 1 (circumradius).
        # If we use circles, we guarantee disjointness and containment if the circle is inside.
        # But a hexagon is smaller than its circumscribed circle.
        # If we pack circles, we get a valid hexagon packing (since hexagon < circle).
        # The score will be worse, but it's a valid lower bound on the difficulty.
        # Let's try packing CIRCLES of radius 1 (diameter 2).
        # Containment: Distance from center to circle center + 1 <= D_outer_apothem? No.
        # Distance from center to circle center + 1 <= D_outer_circumradius?
        # The outer hexagon contains the circle if the distance from the center to the closest side (apothem) is >= distance(center, circle_center) + 1?
        # No, the circle must be inside. The distance from the center to the circle center + radius <= apothem_outer.
        # So D_outer_apothem >= dist + 1.
        
        # Disjointness: dist(center_i, center_j) >= 2.
        
        # This is a classic circle packing problem!
        # Let's optimize circle packing of 11 circles of radius 1 in a hexagon.
        # This is much easier to handle with smooth penalties.
        
        # Objective: Minimize D_outer_apothem (which maps to S) + penalties for circle overlaps and containment.
        
        # Required D_outer_apothem for containment:
        # max_j (dist(C, P_j) + 1)
        dists = np.linalg.norm(rel_pts, axis=1)
        required_D = np.max(dists + 1.0)
        
        S = (required_D + APOTHEM) / APOTHEM
        
        # Penalties
        # 1. Overlap penalty
        # Pairwise distances
        # (11, 11) matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        # We want dist_matrix[i,j] >= 2 for i != j
        # Penalty = sum(max(0, 2 - dist)^2)
        # Exclude diagonal
        mask = ~np.eye(11, dtype=bool)
        overlaps = np.maximum(0, 2.0 - dist_matrix)
        overlap_penalty = np.sum(overlaps[mask] ** 2)
        
        # 2. Containment penalty (already handled by required_D, but we want to minimize S)
        # We are minimizing S directly.
        
        # Total cost
        cost = S + 100.0 * overlap_penalty
        
        return cost

    best_S = float('inf')
    best_params = None

    # Run optimization for each initial set
    for i, pts_init in enumerate(all_initial_sets):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Initial guess for center and rotation
        cx_init = np.mean(pts_init[:, 0])
        cy_init = np.mean(pts_init[:, 1])
        beta_init = 0.0
        
        x0 = np.concatenate([pts_init.flatten(), [cx_init, cy_init, beta_init]])
        
        # Nelder-Mead is good for small dimensions (25D)
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_S:
                # Check if valid (no overlaps)
                # Extract params
                p = res.x
                pts_opt = p[:22].reshape(11, 2)
                cx_opt = p[22]
                cy_opt = p[23]
                beta_opt = p[24]
                
                # Verify overlaps
                diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
                dist_matrix = np.linalg.norm(diff, axis=2)
                mask = ~np.eye(11, dtype=bool)
                min_dist = np.min(dist_matrix[mask])
                
                if min_dist >= 1.99: # Tolerance
                    # Calculate exact S for this configuration
                    # Using the circle approximation for S is safe (hexagon is inside circle)
                    # But we can calculate the exact hexagon S for better score.
                    # However, if we use circle S, it's a valid upper bound on the true S.
                    # We want to MINIMIZE the score.
                    # If we report the circle-based S, it might be larger than the true hexagon S.
                    # But it's a valid packing (hexagons are inside circles, circles inside hexagon).
                    # So the reported S is valid.
                    
                    # To get a better score, we should calculate the true hexagon S.
                    # Let's do that.
                    
                    # True S calculation
                    normals = []
                    for k in range(6):
                        ang = beta_opt + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    normals = np.array(normals)
                    
                    # For each inner hexagon, we need to check if it's inside the outer.
                    # This is equivalent to: For each outer normal n, and each inner hexagon vertex v,
                    # (v + center_inner) . n <= D_outer + center_outer . n ?
                    # Actually, the condition is:
                    # max_{v in inner_hex} ((v + c_i) . n) <= c_outer . n + D_outer
                    # for all n, i.
                    
                    # Inner hexagon vertices (relative to center)
                    # They are rotated by 0 (assumed)
                    # Vertices: (1, 0), (0.5, sqrt(3)/2), ...
                    # We need the max projection of the inner hexagon in direction n.
                    # This is the support function h(n).
                    # For a regular hexagon, h(n) is periodic.
                    # We can compute it by projecting the 6 vertices.
                    
                    inner_verts = unit_hex_verts
                    # Project inner vertices onto normals
                    # (6, 6)
                    inner_projs = inner_verts @ normals.T
                    max_inner_projs = np.max(inner_projs, axis=0) # (6,)
                    
                    # For each inner hexagon i, the max projection in direction k is:
                    # (c_i . n_k) + max_inner_projs[k]
                    
                    c_i_projs = pts_opt @ normals.T # (11, 6)
                    total_max_projs = c_i_projs + max_inner_projs[np.newaxis, :] # (11, 6)
                    max_over_all = np.max(total_max_projs, axis=0) # (6,)
                    
                    # Similarly for min
                    min_inner_projs = np.min(inner_projs, axis=0)
                    total_min_projs = c_i_projs + min_inner_projs[np.newaxis, :]
                    min_over_all = np.min(total_min_projs, axis=0)
                    
                    # The outer hexagon centered at c_outer with rotation beta has "width" 2*D in each normal direction.
                    # The projection of the outer hexagon onto n_k is [c_outer . n_k - D, c_outer . n_k + D]
                    # We need:
                    # max_over_all[k] <= c_outer . n_k + D
                    # min_over_all[k] >= c_outer . n_k - D
                    
                    c_outer_projs = np.array([cx_opt, cy_opt]) @ normals.T # (6,)
                    
                    # Required D_k = max( max_over_all[k] - c_outer_projs[k], c_outer_projs[k] - min_over_all[k] )
                    D_required = np.max(np.abs(total_max_projs - c_outer_projs[np.newaxis, :]), axis=0)
                    D_required = np.maximum(D_required, np.abs(total_min_projs - c_outer_projs[np.newaxis, :]))
                    D_final = np.max(D_required)
                    
                    S_true = (D_final + APOTHEM) / APOTHEM
                    
                    if S_true < best_S:
                        best_S = S_true
                        best_params = (pts_opt, cx_opt, cy_opt, beta_opt)
                        
        except Exception:
            continue

    if best_params is None:
        # Fallback to initial best
        for i, pts_init in enumerate(all_initial_sets):
            cx_init = np.mean(pts_init[:, 0])
            cy_init = np.mean(pts_init[:, 1])
            beta_init = 0.0
            
            normals = []
            for k in range(6):
                ang = beta_init + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            normals = np.array(normals)
            
            inner_verts = unit_hex_verts
            inner_projs = inner_verts @ normals.T
            max_inner_projs = np.max(inner_projs, axis=0)
            min_inner_projs = np.min(inner_projs, axis=0)
            
            c_i_projs = pts_init @ normals.T
            total_max_projs = c_i_projs + max_inner_projs[np.newaxis, :]
            total_min_projs = c_i_projs + min_inner_projs[np.newaxis, :]
            
            c_outer_projs = np.array([cx_init, cy_init]) @ normals.T
            
            D_required = np.max(np.abs(total_max_projs - c_outer_projs[np.newaxis, :]), axis=0)
            D_required = np.maximum(D_required, np.abs(total_min_projs - c_outer_projs[np.newaxis, :]))
            D_final = np.max(D_required)
            
            S_true = (D_final + APOTHEM) / APOTHEM
            
            if S_true < best_S:
                best_S = S_true
                best_params = (pts_init, cx_init, cy_init, beta_init)

    if best_params is None:
        # Absolute fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    pts_opt, cx_opt, cy_opt, beta_opt = best_params
    
    # Scale up slightly to avoid boundary precision issues?
    # The evaluator checks validity. If we are exactly on the boundary, it might fail due to floating point.
    # We should scale the OUTER hexagon slightly larger, or move points slightly inward.
    # Moving points inward is safer.
    
    # Let's recalculate the exact S for the returned configuration to be sure.
    # And apply a tiny scale factor to the inner points towards the center.
    
    # Actually, the best strategy is to return the configuration as is, but ensure the outer side is slightly larger.
    # The score is the outer side length.
    # If we increase the outer side, the score increases.
    # We want the MINIMUM valid side.
    # So we should return the calculated S_true.
    # But if S_true is invalid due to float error, we might get 0 score.
    # Let's add a tiny epsilon to S_true.
    
    final_S = best_S * (1.0 + 1e-9)
    
    inner = []
    for x, y in pts_opt:
        inner.append([float(x), float(y), float(0.0)]) # Angle 0 for inner hexagons
        
    center = [float(cx_opt), float(cy_opt)]
    side = float(final_S)
    rotation = float(math.degrees(beta_opt))
    
    return inner, center, side, rotation
