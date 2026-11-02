import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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

    def compute_min_side(hexes, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, given orientation)
        that contains all given hexagons.
        """
        # The outer hexagon has fixed orientation (angle 0 by default, but can be rotated).
        # Its supporting lines are defined by the 6 outward normals of a hexagon with side 1, angle 0.
        # For a regular hexagon, the outward normals are perpendicular to the sides.
        # Vertices are at angles: angle + 0, 60, ..., 300.
        # Sides are between vertices. Normal to side connecting v_i and v_{i+1} is at angle + 30 + i*60.
        normal_angles = outer_angle + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # For a regular hexagon of side s, the distance from center to side (apothem) is s * sqrt(3) / 2.
        # So max_proj = s * sqrt(3) / 2
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_objective_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2, outer_angle]
        Generates 12 hexagons using 6-fold symmetry.
        Returns the minimum outer side length required to contain them + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2, outer_angle = params
        
        # Generate the 12 hexagons
        # Base hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        penalty = 0.0
        
        # Check overlaps
        # Only check pairs that are potentially close to save time?
        # With 12 hexes, 66 pairs. SAT is fast.
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
        
        if penalty > 1e-6:
            # If there is overlap, return a large value to discourage this configuration
            # We can still return the side length plus penalty, but a large penalty ensures invalid configs are avoided
            # Actually, if penalty is large, the objective is large.
            pass
            
        # Compute the minimum side length required
        s = compute_min_side(hexes, outer_angle)
        
        return s + penalty

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons and outer angle
    # x, y in [-5, 5]
    # angle in [0, pi/3]
    # outer_angle in [0, pi/6] (due to symmetry of hexagon, 0 to 30 deg is enough)
    bounds = [
        (-5.0, 5.0),  # x1
        (-5.0, 5.0),  # y1
        (0.0, np.pi/3), # a1
        (-5.0, 5.0),  # x2
        (-5.0, 5.0),  # y2
        (0.0, np.pi/3), # a2
        (0.0, np.pi/6)   # outer_angle
    ]
    
    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search
    try:
        res = differential_evolution(
            compute_objective_symmetric,
            bounds,
            maxiter=300,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Refinement: Try a few local minimizations from good symmetric starts
    # Start with a known good configuration heuristic:
    # Hexagons arranged in a ring.
    # Let's try a few random symmetric starts and polish them.
    
    if time.time() < deadline - 10:
        # Try a few random starts
        for _ in range(10):
            p = np.array([
                np.random.uniform(2, 4), np.random.uniform(-1, 1), np.random.uniform(0, np.pi/3),
                np.random.uniform(3, 5), np.random.uniform(-1, 1), np.random.uniform(0, np.pi/3),
                np.random.uniform(0, np.pi/6)
            ])
            val = compute_objective_symmetric(p)
            if val < best_s:
                best_s = val
                best_params = p
            try:
                res_local = minimize(
                    compute_objective_symmetric,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass
            if time.time() > deadline - 5:
                break

    if best_params is None:
        # Fallback: Simple grid or known approximate
        # Try a few random symmetric starts
        for _ in range(5):
            p = np.array([
                np.random.uniform(1, 3), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                np.random.uniform(2, 4), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                np.random.uniform(0, np.pi/6)
            ])
            val = compute_objective_symmetric(p)
            if val < best_s:
                best_s = val
                best_params = p
            try:
                res_local = minimize(
                    compute_objective_symmetric,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass
            if time.time() > deadline - 5:
                break
                
    x1, y1, a1, x2, y2, a2, outer_angle = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        # Rotate (x1, y1)
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        # Rotate (x2, y2)
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
        inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
        
    outer_center = (0.0, 0.0)
    # Recompute the exact side length for the final configuration
    # We need to regenerate the hexes to compute the exact min side
    base1 = hex_vertices(x1, y1, 1.0, a1)
    base2 = hex_vertices(x2, y2, 1.0, a2)
    hexes = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        rot = np.array([[c, -s_], [s_, c]])
        h1 = base1 @ rot.T
        h2 = base2 @ rot.T
        hexes.append(h1)
        hexes.append(h2)
        
    outer_side = compute_min_side(hexes, outer_angle)
    outer_angle_deg = np.degrees(outer_angle)
    
    return inner, outer_center, outer_side, outer_angle_deg
