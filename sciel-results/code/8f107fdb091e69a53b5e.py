import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon (CCW order)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        # Get edges and normals
        def get_normals(verts):
            edges = np.roll(verts, -1, axis=0) - verts
            normals = np.column_stack([edges[:, 1], -edges[:, 0]])
            norms = np.linalg.norm(normals, axis=1)
            normals = normals / (norms[:, np.newaxis] + 1e-15)
            return normals

        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return min_overlap

    def compute_min_side(hexes, outer_angle_rad):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, given angle)
        that contains all given hexagons.
        """
        normal_angles = outer_angle_rad + np.radians(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def objective(params):
        """
        params: [x1, y1, ..., x12, y12, outer_angle] (25 params for fixed angle case)
        or [x1, y1, a1, ..., x12, y12, a12, outer_angle] (37 params for full case)
        We will implement the 25-param version first for speed, then 37-param.
        Here we implement the 25-param version (fixed inner angles = 0).
        """
        if time.time() > deadline:
            return 1e10

        n_hex = 12
        positions = params[:24].reshape(n_hex, 2)
        outer_angle = params[24]
        
        # Center the configuration
        centroid = np.mean(positions, axis=0)
        positions = positions - centroid
        
        hexes = []
        for i in range(n_hex):
            h = hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
            hexes.append(h)
            
        # Overlap Penalty
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * overlap * 100.0

        if penalty > 1e-6:
            return 1e10 + penalty

        s = compute_min_side(hexes, outer_angle)
        return s

    def objective_full(params):
        """
        params: [x1, y1, a1, ..., x12, y12, a12, outer_angle] (37 params)
        """
        if time.time() > deadline:
            return 1e10

        n_hex = 12
        data = params[:36].reshape(n_hex, 3)
        outer_angle = params[36]
        
        positions = data[:, :2]
        angles = data[:, 2]
        
        # Center the configuration
        centroid = np.mean(positions, axis=0)
        positions = positions - centroid
        
        hexes = []
        for i in range(n_hex):
            h = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            hexes.append(h)
            
        # Overlap Penalty
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * overlap * 100.0

        if penalty > 1e-6:
            return 1e10 + penalty

        s = compute_min_side(hexes, outer_angle)
        return s

    # --- Initial Topology ---
    # Honeycomb cluster
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    base_positions = np.array([axial_to_cartesian(q, r) for q, r in lattice_coords])
    base_positions -= np.mean(base_positions, axis=0)
    
    initial_params_25 = np.concatenate([base_positions.flatten(), [0.0]])
    
    best_s = np.inf
    best_params_25 = initial_params_25
    best_params_37 = None
    
    # Phase 1: Optimize positions (fixed angles)
    # Using Differential Evolution
    bounds_25 = [(-5, 5)] * 24 + [(-np.pi/6, np.pi/6)]
    
    if time.time() < deadline - 40:
        try:
            res_de = differential_evolution(
                objective,
                bounds_25,
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                polish=False
            )
            if res_de.fun < best_s:
                best_s = res_de.fun
                best_params_25 = res_de.x
        except Exception:
            pass

    # Polish Phase 1
    if time.time() < deadline - 20:
        try:
            res_nm = minimize(
                objective,
                best_params_25,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nm.fun < best_s:
                best_s = res_nm.fun
                best_params_25 = res_nm.x
        except Exception:
            pass

    # Phase 2: Optimize angles as well
    # Initialize from Phase 1 result
    init_37 = np.zeros(37)
    for i in range(12):
        init_37[i*3] = best_params_25[i*2]
        init_37[i*3+1] = best_params_25[i*2+1]
        init_37[i*3+2] = 0.0 # Initial angle 0
    init_37[36] = best_params_25[24]
    
    best_s_37 = objective_full(init_37)
    best_params_37 = init_37
    
    if time.time() < deadline - 10:
        try:
            # Use Nelder-Mead for the full 37-dim problem, starting from the 25-dim solution
            res_nm_full = minimize(
                objective_full,
                init_37,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nm_full.fun < best_s_37:
                best_s_37 = res_nm_full.fun
                best_params_37 = res_nm_full.x
        except Exception:
            pass

    # Determine final best
    if best_s_37 < best_s:
        final_params = best_params_37
        final_s = best_s_37
        use_full = True
    else:
        final_params = best_params_25
        final_s = best_s
        use_full = False

    # Construct final hexes
    if use_full:
        n_hex = 12
        data = final_params[:36].reshape(n_hex, 3)
        outer_angle = final_params[36]
        positions = data[:, :2]
        angles = data[:, 2]
        
        centroid = np.mean(positions, axis=0)
        positions = positions - centroid
        
        hexes = []
        for i in range(n_hex):
            h = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            hexes.append(h)
    else:
        n_hex = 12
        positions = final_params[:24].reshape(n_hex, 2)
        outer_angle = final_params[24]
        
        centroid = np.mean(positions, axis=0)
        positions = positions - centroid
        
        hexes = []
        for i in range(n_hex):
            h = hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
            hexes.append(h)

    # Verify validity
    valid = True
    for i in range(12):
        for j in range(i + 1, 12):
            min1 = hexes[i].min(axis=0)
            max1 = hexes[i].max(axis=0)
            min2 = hexes[j].min(axis=0)
            max2 = hexes[j].max(axis=0)
            
            if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                continue
                
            overlap = sat_overlap(hexes[i], hexes[j])
            if overlap > 1e-6:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to a known valid configuration (scaled up)
        # Just use the base positions with a larger scale to ensure no overlap
        # But we need to return a valid solution.
        # Let's just return the base honeycomb with a large enough outer hex.
        # Base honeycomb fits in a hexagon of side ~ 2*sqrt(3) + something?
        # Actually, just return the base configuration and let the outer hex be large.
        # The score will be worse, but it's valid.
        hexes = []
        for pos in base_positions:
            h = hex_vertices(pos[0], pos[1], 1.0, 0.0)
            hexes.append(h)
        outer_angle = 0.0
        final_s = compute_min_side(hexes, outer_angle)
    else:
        final_s = compute_min_side(hexes, outer_angle)

    inner = []
    for i in range(12):
        cx, cy = np.mean(hexes[i], axis=0)
        if use_full:
            angle_deg = np.degrees(angles[i])
        else:
            angle_deg = 0.0
        inner.append([cx, cy, angle_deg])
        
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(outer_angle)
    
    return inner, outer_center, outer_side, outer_angle_deg
