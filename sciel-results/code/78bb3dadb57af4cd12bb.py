import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 140.0
    safety_margin = 10.0

    # Geometry constants
    d = math.sqrt(3)
    spacing = 2 * d  # Distance between centers of touching unit hexagons (apothem*2)
    # Note: For unit side length hexagons, apothem is sqrt(3)/2.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait, let's re-verify.
    # Side length s=1.
    # Apothem a = s * sqrt(3) / 2 = sqrt(3) / 2.
    # Distance between parallel sides is 2a = sqrt(3).
    # If two hexagons touch side-to-side, their centers are 2a apart? No.
    # Let's visualize. Two hexagons sharing an edge. The distance from center to edge is a.
    # So center-to-center distance is 2a = sqrt(3).
    # The current code uses spacing = 2 * d = 2 * sqrt(3). This is WRONG for touching side-to-side.
    # 2 * sqrt(3) is the distance between opposite vertices (diameter)? No, diameter is 2s = 2.
    # Distance between opposite sides is sqrt(3).
    # Distance between opposite vertices is 2.
    # If hexagons are packed in a lattice, the fundamental translation vectors are length sqrt(3)?
    # Let's check standard hexagonal packing.
    # Centers form a triangular lattice. Distance between nearest neighbors is 2 * apothem = sqrt(3).
    # So spacing should be sqrt(3).
    # The current program uses spacing = 2 * sqrt(3). This is double the touching distance.
    # This means the current program has large gaps between hexagons, which is why the score is poor (large S).
    # Let's correct this fundamental geometry error first.
    
    spacing = d # sqrt(3)

    # Relative vertices of a unit hexagon (pointy-top, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    verts_rel = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
    ])

    # Generate base centers: Center (0,0) + Ring 1 (6 hexagons)
    base_centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = spacing * math.cos(angle)
        y = spacing * math.sin(angle)
        base_centers.append((x, y))
    
    # Generate Ring 2 centers (12 hexagons)
    # Axial coordinates (q, r) where max(|q|, |r|, |q+r|) == 2
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = spacing * (q + r / 2.0)
                y = spacing * (d / 2.0) * r
                ring2_centers.append((x, y))
    
    # Ensure we have 12 ring 2 centers
    if len(ring2_centers) != 12:
        for i in range(12):
            angle = i * math.pi / 6
            ring2_centers.append((2 * spacing * math.cos(angle), 2 * spacing * math.sin(angle)))

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def sat_overlap_depth(v1, v2):
        """Return maximum overlap depth (positive if overlapping, negative if separated)."""
        max_overlap = 0.0
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                max_overlap = max(max_overlap, overlap)
        return max_overlap

    def compute_S_from_points(points, theta, center=(0.0, 0.0)):
        """
        Compute minimal side length S of a regular hexagon with rotation theta
        centered at 'center' that encloses the given points.
        """
        # Translate points to center
        pts = points - np.array(center)
        
        # Normals of the hexagon sides
        # The sides are perpendicular to the normals.
        # For a regular hexagon, the normals point to the midpoints of the sides.
        # If rotation is theta, the first normal is at angle theta + pi/6?
        # Let's check.
        # Vertex at angle 0. Side midpoint at pi/6.
        # So normals are at theta + pi/6 + k*pi/3.
        k_angles = theta + math.pi/6 + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        # Project points onto normals
        dots = np.dot(pts, normals.T)
        
        # Extents: 6
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)
        
        # S = max_extent / sqrt(3)
        # Wait. The width of a hexagon across sides is 2 * apothem = sqrt(3) * S.
        # So max_extent = sqrt(3) * S.
        # S = max_extent / sqrt(3).
        return max_extent / math.sqrt(3)

    def compute_S_and_theta(params, theta, center=(0.0, 0.0)):
        """
        Compute S for given params (x,y,angle for 11 hexagons) and outer theta.
        params: array of size 33
        """
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        return compute_S_from_points(all_pts, theta, center)

    def objective(params, theta, center=(0.0, 0.0)):
        if time.time() > deadline - safety_margin:
            return 1e10
            
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                overlap = sat_overlap_depth(verts_list[i], verts_list[j])
                if overlap > 0:
                    penalty += overlap * overlap * 1000.0
        
        S = compute_S_and_theta(params, theta, center)
        return S + penalty

    # Step 1: Enumerate combinations and find best rigid configuration
    # We fix the inner hexagons in a rigid lattice (angle 0) and optimize the outer container.
    # This is a structural change: treat inner as rigid, optimize outer.
    # Then, use this as a starting point for a full optimization.
    
    best_S_rigid = float('inf')
    best_params_rigid = None
    best_theta_rigid = 0.0
    top_candidates = [] # List of (S, combo, theta)

    all_combinations = list(itertools.combinations(range(12), 4))
    
    # Pre-compute base vertices for speed
    base_verts_list = [get_hex_verts(cx, cy, 0.0) for cx, cy in base_centers]
    base_pts = np.vstack(base_verts_list)

    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
            
        # Get ring 2 centers for this combo
        r2_centers = [ring2_centers[i] for i in combo]
        
        # Compute vertices for these 4 hexagons (angle=0)
        r2_verts_list = [get_hex_verts(cx, cy, 0.0) for cx, cy in r2_centers]
        r2_pts = np.vstack(r2_verts_list)
        
        # Combine all points
        all_pts = np.vstack([base_pts, r2_pts])
        
        # Sweep theta for this configuration
        # High resolution grid search for theta
        thetas = np.linspace(0, math.pi / 3, 200)
        
        min_S = float('inf')
        best_theta = 0.0
        
        for theta in thetas:
            S_val = compute_S_from_points(all_pts, theta, (0.0, 0.0))
            if S_val < min_S:
                min_S = S_val
                best_theta = theta
        
        if min_S < best_S_rigid:
            # Update top candidates
            if len(top_candidates) < 10:
                top_candidates.append((min_S, combo, best_theta))
            else:
                # Replace worst if better
                worst_idx = max(range(len(top_candidates)), key=lambda i: top_candidates[i][0])
                if min_S < top_candidates[worst_idx][0]:
                    top_candidates[worst_idx] = (min_S, combo, best_theta)
                    
            best_S_rigid = min_S
            best_theta_rigid = best_theta
            # Construct params for best rigid config
            final_centers = base_centers + r2_centers
            params = np.zeros(N * 3)
            for i, (cx, cy) in enumerate(final_centers):
                params[3 * i] = cx
                params[3 * i + 1] = cy
                params[3 * i + 2] = 0.0
            best_params_rigid = params

    # Step 2: Refine the top candidates
    # Use the rigid configuration as a starting point for a full optimization
    # allowing inner hexagons to move and rotate, and the outer container to move and rotate.
    # This combines the "rigid lattice" structural insight with the "full optimization" refinement.
    
    best_S_final = best_S_rigid
    best_params_final = best_params_rigid
    best_theta_final = best_theta_rigid
    best_center_final = (0.0, 0.0)
    
    # Sort top candidates by S
    top_candidates.sort(key=lambda x: x[0])
    
    for S_cand, combo, theta_cand in top_candidates[:5]:
        if time.time() > deadline - safety_margin:
            break
            
        # Initialize params from rigid config
        r2_centers = [ring2_centers[i] for i in combo]
        final_centers = base_centers + r2_centers
        init_params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers):
            init_params[3 * i] = cx
            init_params[3 * i + 1] = cy
            init_params[3 * i + 2] = 0.0
            
        # Objective wrapper for scipy
        # x: [inner_params (33), outer_center_x, outer_center_y, outer_theta]
        def obj_wrapper(x):
            p = x[:33]
            ocx = x[33]
            ocy = x[34]
            t = x[35]
            return objective(p, t, (ocx, ocy))
            
        init_x = np.append(init_params, [0.0, 0.0, theta_cand])
        
        try:
            # Use Powell method for robustness
            res = minimize(obj_wrapper, init_x, method='Powell',
                           options={'maxiter': 500, 'maxfev': 5000, 'ftol': 1e-5, 'xtol': 1e-5})
            
            if res.fun < best_S_final:
                # Check validity
                p_res = res.x[:33]
                ocx_res = res.x[33]
                ocy_res = res.x[34]
                t_res = res.x[35]
                
                # Calculate actual S and penalty
                n_hex = len(p_res) // 3
                xs = p_res[0::3]
                ys = p_res[1::3]
                thetas = p_res[2::3]
                verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
                
                max_overlap = 0.0
                for i in range(n_hex):
                    for j in range(i + 1, n_hex):
                        ov = sat_overlap_depth(verts_list[i], verts_list[j])
                        if ov > max_overlap:
                            max_overlap = ov
                
                if max_overlap < 1e-6: # Valid
                    all_pts = np.vstack(verts_list)
                    S_actual = compute_S_from_points(all_pts, t_res, (ocx_res, ocy_res))
                    
                    if S_actual < best_S_final:
                        best_S_final = S_actual
                        best_params_final = p_res
                        best_theta_final = t_res
                        best_center_final = (ocx_res, ocy_res)
                        
        except Exception:
            continue

    # Construct final output
    # If refinement failed or didn't improve, use rigid config
    if best_params_final is None:
        # Fallback to first rigid config
        best_params_final = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(base_centers + [ring2_centers[i] for i in range(4)]):
            best_params_final[3 * i] = cx
            best_params_final[3 * i + 1] = cy
            best_params_final[3 * i + 2] = 0.0
        best_theta_final = 0.0
        best_center_final = (0.0, 0.0)
        best_S_final = compute_S_and_theta(best_params_final, best_theta_final, best_center_final)

    inner = []
    for i in range(N):
        x = best_params_final[3 * i]
        y = best_params_final[3 * i + 1]
        angle_deg = math.degrees(best_params_final[3 * i + 2])
        inner.append([x, y, angle_deg])

    return inner, best_center_final, best_S_final, math.degrees(best_theta_final)
