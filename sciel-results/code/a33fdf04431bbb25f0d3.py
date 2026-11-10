import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # 50s margin for final processing

    # Precompute constants
    base_angles_rad = np.deg2rad(np.arange(0, 360, 60))
    base_verts = np.column_stack([np.cos(base_angles_rad), np.sin(base_angles_rad)])
    outer_normal_angles = np.deg2rad(np.arange(30, 360, 60))
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])
    
    # Precompute edge vectors and normals for base hexagon (for speed in separation check)
    # Edges: v[i+1] - v[i]
    base_edges = np.roll(base_verts, -1, axis=0) - base_verts
    # Normals: perpendicular to edge. For CCW polygon, inward normal is (dy, -dx) rotated?
    # Let's check: Edge (1,0)->(0.5, 0.866). vec = (-0.5, 0.866).
    # Inward normal should point towards (0,0).
    # vec = (-0.5, 0.866). Perpendiculars: (0.866, 0.5) and (-0.866, -0.5).
    # (0.866, 0.5) dot (-0.5, 0.866) = 0.
    # Let's just compute normals as (edge_y, -edge_x) normalized.
    base_edge_normals = np.column_stack([base_edges[:, 1], -base_edges[:, 0]])
    base_edge_normal_lens = np.linalg.norm(base_edge_normals, axis=1)
    base_edge_normals_norm = base_edge_normals / base_edge_normal_lens[:, np.newaxis]

    def get_verts_xy(params):
        """Returns (11, 6, 2) vertices array and (11, 2) centers."""
        xy = params[:, :2]
        angles = params[:, 2]
        cos_a = np.cos(np.deg2rad(angles))
        sin_a = np.sin(np.deg2rad(angles))
        
        vx = base_verts[:, 0]
        vy = base_verts[:, 1]
        
        rot_vx = cos_a[:, None] * vx[None, :] - sin_a[:, None] * vy[None, :]
        rot_vy = sin_a[:, None] * vx[None, :] + cos_a[:, None] * vy[None, :]
        
        verts_x = xy[:, 0:1] + rot_vx
        verts_y = xy[:, 1:2] + rot_vy
        verts = np.stack([verts_x, verts_y], axis=-1)
        return verts, xy

    def evaluate(params):
        if time.time() > deadline:
            return 1e10
            
        verts, xy = get_verts_xy(params)
        
        # 1. Containment / Side Length S
        flat_verts = verts.reshape(-1, 2)
        projections = flat_verts @ outer_normals.T
        # The outer hexagon is centered at origin.
        # For a regular hexagon with side S and angle 0, the apothem is S*sqrt(3)/2.
        # The max projection of the inner set onto the outer normals determines the required apothem.
        # Actually, the outer hexagon vertices are at distance S from center.
        # The edge normals point to midpoints.
        # If the max projection on normal n is D, then the distance from center to that side is D.
        # For a regular hexagon, distance from center to side (apothem) a = S * sqrt(3) / 2.
        # So S = 2 * a / sqrt(3).
        max_proj = np.max(np.max(projections, axis=1))
        S = 2.0 * max_proj / np.sqrt(3)
        
        # 2. Separation
        # We need min gap > 1e-6 for all pairs.
        # Gap is the minimum distance between two convex polygons along any normal.
        # If they intersect, gap is negative.
        # We check all 12 normals (6 from hex 1, 6 from hex 2).
        
        min_gap_val = np.inf
        for i in range(11):
            for j in range(i + 1, 11):
                v1 = verts[i]
                v2 = verts[j]
                
                # Normals of v1
                # Rotated base normals
                a1 = np.deg2rad(params[i, 2])
                c1, s1 = np.cos(a1), np.sin(a1)
                R1 = np.array([[c1, -s1], [s1, c1]])
                n1 = base_edge_normals_norm @ R1.T
                
                # Normals of v2
                a2 = np.deg2rad(params[j, 2])
                c2, s2 = np.cos(a2), np.sin(a2)
                R2 = np.array([[c2, -s2], [s2, c2]])
                n2 = base_edge_normals_norm @ R2.T
                
                all_normals = np.vstack([n1, n2])
                
                # Project v1 and v2 onto all_normals
                proj1 = v1 @ all_normals.T
                proj2 = v2 @ all_normals.T
                
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                
                # Gap on each axis
                gaps = np.maximum(min2 - max1, min1 - max2)
                gap = np.min(gaps)
                
                if gap < min_gap_val:
                    min_gap_val = gap
        
        penalty = 0.0
        if min_gap_val < 1e-6:
            # Strong penalty for violation
            penalty = (1e-6 - min_gap_val) ** 2 * 1e5
        return S + penalty

    # Initialization
    # 1 center, 6 ring, 4 outer
    # d = sqrt(3) is distance to ring 1 centers for touching hexagons.
    d1 = np.sqrt(3)
    x0 = np.zeros(11)
    y0 = np.zeros(11)
    a0 = np.zeros(11)
    
    for k in range(6):
        ang = k * 60
        x0[1 + k] = d1 * np.cos(np.deg2rad(ang))
        y0[1 + k] = d1 * np.sin(np.deg2rad(ang))
        
    # 4 outer hexagons.
    # Place them in the "corners" of the hexagonal cluster of 7.
    # The 7-hex cluster has a bounding hexagon of side 2 (in units of hex side length? No, side length of outer hex is determined by positions).
    # Let's place them at distance ~ 2*sqrt(3) at angles 30, 150, 270, 330? Or 0, 90, 180, 270?
    # A common packing for 11 is the 7 cluster + 4 in the gaps between ring 1 and the next ring.
    # Gaps in ring 1 are at 30, 90, 150... no, ring 1 is at 0, 60...
    # Gaps are at 30, 90, 150, 210, 270, 330.
    # Let's pick 4 of them.
    gap_angles = [30, 150, 270, 330]
    d2 = 2.0 * np.sqrt(3) # Approximate distance
    for k, ang in enumerate(gap_angles):
        x0[7 + k] = d2 * np.cos(np.deg2rad(ang))
        y0[7 + k] = d2 * np.sin(np.deg2rad(ang))
        
    params0 = np.column_stack([x0, y0, a0])
    
    bounds = [(-5, 5), (-5, 5), (0, 360)] * 11
    bounds = [tuple(b) for b in bounds]
    
    best_S = 1e10
    best_params = params0
    
    # Run Differential Evolution
    # Use a reasonable population size and iteration count
    try:
        res = differential_evolution(
            evaluate, 
            bounds, 
            seed=42, 
            maxiter=500, 
            popsize=15, 
            tol=1e-4, 
            polish=False,
            init='sobol'
        )
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass

    # Local Refinement with Powell (robust for non-smooth)
    if time.time() < deadline - 20:
        try:
            res_p = minimize(
                evaluate, 
                best_params, 
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-5, 'ftol': 1e-5}
            )
            if res_p.fun < best_S:
                best_S = res_p.fun
                best_params = res_p.x
        except Exception:
            pass

    # One more refinement with Nelder-Mead if time permits
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                evaluate, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_nm.fun < best_S:
                best_S = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    # Final Validation and Extraction
    final_S_penalized = evaluate(best_params)
    
    xy = best_params[:, :2]
    angles = best_params[:, 2]
    verts, _ = get_verts_xy(best_params)
    flat_verts = verts.reshape(-1, 2)
    projections = flat_verts @ outer_normals.T
    max_proj = np.max(np.max(projections, axis=1))
    pure_S = 2.0 * max_proj / np.sqrt(3)
    
    # Check validity strictly
    valid = True
    if not np.isfinite(pure_S):
        valid = False
    if pure_S < np.sqrt(11):
        valid = False
        
    # Check separation strictly
    min_gap_val = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = verts[i]
            v2 = verts[j]
            a1 = np.deg2rad(params[i, 2])
            c1, s1 = np.cos(a1), np.sin(a1)
            R1 = np.array([[c1, -s1], [s1, c1]])
            n1 = base_edge_normals_norm @ R1.T
            a2 = np.deg2rad(params[j, 2])
            c2, s2 = np.cos(a2), np.sin(a2)
            R2 = np.array([[c2, -s2], [s2, c2]])
            n2 = base_edge_normals_norm @ R2.T
            all_normals = np.vstack([n1, n2])
            proj1 = v1 @ all_normals.T
            proj2 = v2 @ all_normals.T
            min1 = np.min(proj1, axis=1)
            max1 = np.max(proj1, axis=1)
            min2 = np.min(proj2, axis=1)
            max2 = np.max(proj2, axis=1)
            gaps = np.maximum(min2 - max1, min1 - max2)
            gap = np.min(gaps)
            if gap < min_gap_val:
                min_gap_val = gap
                
    if min_gap_val < 1e-6:
        valid = False
        
    if not valid:
        # Fallback
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(pure_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
