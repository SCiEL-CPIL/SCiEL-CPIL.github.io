import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 570
    
    # Define the rigid cluster of 12 hexagons.
    # We use a "compact" arrangement.
    # Standard hexagonal lattice coordinates (q, r) for hexagons of side 1.
    # Distance between centers of touching hexagons is sqrt(3).
    # Basis vectors:
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 1.5)
    
    h = np.sqrt(3) / 2
    # Lattice vectors
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    # We want a packing that is roughly circular/symmetric to fit in a hexagon.
    # A common compact shape for 12 hexagons is a "flower" of 7 (1+6) plus 5 more.
    # Or a 3-4-3-2 strip? 
    # Let's try the "center + 6 + 5" arrangement.
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3).
    #   (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in lattice coords.
    # Ring 2: We need 5 more.
    #   The "corners" of the hexagon of side 2 are at distance 2*sqrt(3).
    #   Lattice coords for ring 2 (distance 2):
    #   (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    #   Let's pick 5 that create a roughly convex hull.
    #   (2,0), (1,1), (0,2), (-1,1) is 4. We need 1 more. (-2,0)? Or (1,-1)?
    #   Let's try a symmetric set.
    #   Actually, a very compact shape is the "hexagon of side 2" minus 3 corner hexagons.
    #   Total hexes in side 2 hexagon = 1 + 6 + 12 = 19.
    #   We need 12. So we remove 7.
    #   If we remove 3 corners and 4 others?
    
    # Let's define specific lattice coordinates (i, j) such that pos = i*v1 + j*v2.
    # We want to minimize the circumradius or bounding box.
    
    # Configuration A: Center + 6 + 5 (asymmetric ring 2)
    coords_A = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 1), (-1, -1)
    ]
    
    # Configuration B: 3-4-3-2 strip? No, let's try a different compact cluster.
    # How about a 2x2x3 block?
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2), (2,2)
    # (0,3), (1,3), (2,3) -> 12 hexes. This is a parallelogram.
    coords_B = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    
    # Configuration C: "Flower" 1+6+5 but different 5.
    # Let's use the 5 that are "closest" to the center in the next ring?
    # Ring 2 coords:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Let's pick (2,0), (1,1), (-1,2), (-2,0), (0,-2) -> 5 points.
    coords_C = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 0), (0, -2)
    ]
    
    configs = [coords_A, coords_B, coords_C]
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    def evaluate_cluster(coords, cluster_rot_deg, outer_rot_deg):
        # Build positions
        positions = np.zeros((12, 2))
        for i, (q, r) in enumerate(coords):
            pos = q * v1 + r * v2
            # Rotate cluster
            rot = np.deg2rad(cluster_rot_deg)
            R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            positions[i] = R @ pos
        
        # All inner hexagons have angle 0 relative to cluster frame?
        # Wait, if we rotate the cluster, do the hexagons rotate too?
        # Yes, for a rigid cluster, the orientation of each hexagon is fixed relative to the cluster.
        # If the cluster is rotated by `cluster_rot_deg`, each hexagon is also rotated by `cluster_rot_deg`.
        # So inner angle = cluster_rot_deg.
        
        inner_angle = cluster_rot_deg
        
        s = _min_outer_side(positions, outer_rot_deg)
        
        # Construct inner array for validity check
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = inner_angle
        
        outer_candidate = np.array([0, 0, outer_rot_deg])
        
        if _check_validity(inner_candidate, outer_candidate, s):
            score = 1.0 / s
            return score, inner_candidate, outer_candidate, s
        else:
            return 0.0, None, None, None

    # We search over cluster_rot (0 to 60) and outer_rot (0 to 60).
    # Due to symmetry, we can restrict ranges, but let's be safe.
    
    # Phase 1: Coarse Grid Search
    # 120 steps in 60 degrees = 0.5 deg resolution.
    n_steps = 120
    rot_vals = np.linspace(0, 60, n_steps, endpoint=False)
    
    for config_idx, coords in enumerate(configs):
        if time.time() > deadline:
            break
            
        for cr in rot_vals:
            for orr in rot_vals:
                score, inner_c, outer_c, s_c = evaluate_cluster(coords, cr, orr)
                if score > best_score:
                    best_score = score
                    best_inner = inner_c
                    best_outer = outer_c
                    best_s = s_c
                    # Optional: local refinement around this point
                    # But let's finish grid first to find the best region
    
    # Phase 2: Local Refinement using Nelder-Mead on the best 3 parameters
    # We need to map back to the specific config that gave the best score.
    # Let's re-run the best grid point to know which config it was.
    # Actually, we can just refine the angles for the config that gave the best score.
    # We didn't store which config. Let's store it.
    
    # Let's restart the tracking properly.
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    best_config_idx = 0
    best_cr = 0
    best_orr = 0
    
    for config_idx, coords in enumerate(configs):
        if time.time() > deadline:
            break
            
        for cr in rot_vals:
            for orr in rot_vals:
                score, inner_c, outer_c, s_c = evaluate_cluster(coords, cr, orr)
                if score > best_score:
                    best_score = score
                    best_inner = inner_c
                    best_outer = outer_c
                    best_s = s_c
                    best_config_idx = config_idx
                    best_cr = cr
                    best_orr = orr
    
    # Refine the best candidate
    if best_inner is not None:
        coords_best = configs[best_config_idx]
        
        def objective(x):
            cr, orr = x
            score, _, _, _ = evaluate_cluster(coords_best, cr, orr)
            return -score
        
        # Multi-start refinement from best grid point and its neighbors
        x0 = np.array([best_cr, best_orr])
        
        # Try a few starts near the best
        starts = [
            x0,
            x0 + np.array([0.5, 0]),
            x0 + np.array([0, 0.5]),
            x0 + np.array([0.5, 0.5]),
            x0 + np.array([-0.5, 0]),
            x0 + np.array([0, -0.5]),
        ]
        
        for start in starts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, start, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < 0: # Valid solution found (score > 0)
                    cr_opt, orr_opt = res.x
                    score_opt, inner_opt, outer_opt, s_opt = evaluate_cluster(coords_best, cr_opt, orr_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_inner = inner_opt
                        best_outer = outer_opt
                        best_s = s_opt
            except Exception:
                pass
    
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback: standard hexagonal packing
        h = np.sqrt(3)/2
        inner_fallback = np.array([
            [0, 0, 0],
            [2*h, 0, 0],
            [h, 1.5, 0],
            [-h, 1.5, 0],
            [-2*h, 0, 0],
            [-h, -1.5, 0],
            [h, -1.5, 0],
            [3*h, 1.5, 0],
            [0, 3, 0],
            [-3*h, 1.5, 0],
            [-3*h, -1.5, 0],
            [3*h, -1.5, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
