import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
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

    def compute_objective(params):
        """
        params: 12 * [x, y, rot] + [outer_cx, outer_cy, outer_rot]
        Returns the side length of the outer hexagon + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        # Unpack inner hexagons
        inner_hexes = []
        for i in range(12):
            x = params[3*i]
            y = params[3*i + 1]
            rot = params[3*i + 2]
            inner_hexes.append(hex_vertices(x, y, 1.0, rot))
            
        # Unpack outer hexagon params
        outer_cx = params[36]
        outer_cy = params[37]
        outer_rot = params[38]
        
        # Check overlaps between inner hexagons
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = inner_hexes[i].min(axis=0)
                max1 = inner_hexes[i].max(axis=0)
                min2 = inner_hexes[j].min(axis=0)
                max2 = inner_hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(inner_hexes[i], inner_hexes[j])
                if overlap > 0:
                    # Strong quadratic penalty to prevent overlaps
                    penalty += overlap * overlap * 1000.0 + overlap * 100.0
                    
        if penalty > 0:
            return 10.0 + penalty # Large value if overlapping

        # Calculate minimum side length of outer hexagon containing all inner vertices
        # The outer hexagon is defined by center (outer_cx, outer_cy), side s, rotation outer_rot.
        # We want to find the minimum s such that all inner vertices are inside.
        # For a regular hexagon with side s, the apothem is a = s * sqrt(3) / 2.
        # The distance from the center to each side is a.
        # The condition for a point P to be inside is that its projection onto each of the 6 outward normals
        # is <= a.
        
        # Normals of the outer hexagon
        normal_angles = outer_rot + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Translate inner vertices to be relative to outer center
        all_verts = np.vstack(inner_hexes)
        rel_verts = all_verts - np.array([outer_cx, outer_cy])
        
        # Project relative vertices onto normals
        proj = rel_verts @ normals.T
        
        # The maximum projection in each normal direction gives the required apothem for that direction
        max_projs = proj.max(axis=0)
        
        # The required apothem is the max of these projections
        required_apothem = np.max(max_projs)
        
        # Side length s = 2 * apothem / sqrt(3)
        s = 2.0 * required_apothem / np.sqrt(3.0)
        
        return s

    # --- Optimization ---
    
    # Bounds
    # Inner hexagons: x, y in [-5, 5], rot in [0, pi/3]
    # Outer hexagon: cx, cy in [-5, 5], rot in [0, pi/3]
    bounds = []
    for i in range(12):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, np.pi/3)) # rot
    bounds.append((-5.0, 5.0)) # outer_cx
    bounds.append((-5.0, 5.0)) # outer_cy
    bounds.append((0.0, np.pi/3)) # outer_rot
    
    best_s = np.inf
    best_params = None
    
    # Phase 1: Differential Evolution
    # 39 dimensions is high for DE, but let's try with a reasonable population size
    try:
        res_de = differential_evolution(
            compute_objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=False
        )
        if res_de.fun < best_s:
            best_s = res_de.fun
            best_params = res_de.x.copy()
    except Exception:
        pass

    # Phase 2: Basin Hopping with L-BFGS-B
    if best_params is not None:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=20,
                T=0.5,
                stepsize=0.5,
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x.copy()
        except Exception:
            pass

    # Phase 3: Final Polish
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_polish = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        # Create a rough initial guess
        params = np.zeros(39)
        # Place 12 hexagons in a grid-like structure
        # Just a simple fallback
        for i in range(12):
            params[3*i] = (i % 4) * 2.0 - 3.0
            params[3*i + 1] = (i // 4) * 2.0 - 1.0
            params[3*i + 2] = 0.0
        params[36] = 0.0
        params[37] = 0.0
        params[38] = 0.0
        best_params = params
        best_s = compute_objective(best_params)

    # Extract results
    inner = []
    for i in range(12):
        x = float(best_params[3*i])
        y = float(best_params[3*i + 1])
        rot_deg = float(np.degrees(best_params[3*i + 2]))
        inner.append([x, y, rot_deg])
        
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_side = float(best_s)
    outer_angle_deg = float(np.degrees(best_params[38]))

    return inner, outer_center, outer_side, outer_angle_deg
