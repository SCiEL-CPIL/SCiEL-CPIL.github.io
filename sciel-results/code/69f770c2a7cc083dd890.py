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

    def inside_violation(inner_verts, outer_verts):
        """Returns max distance inner sticks out of outer (0 if inside)."""
        outer_normals = get_normals(outer_verts)
        
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def compute_objective_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2, s]
        Generates 12 hexagons using 6-fold symmetry.
        Returns s + penalty.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2, s = params
        
        if s < 1.5:
            return 1e6
            
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
        
        # Outer hexagon
        outer_verts = hex_vertices(0, 0, s, 0.0)
        
        penalty = 0.0
        
        # Check containment
        for h in hexes:
            viol = inside_violation(h, outer_verts)
            penalty += viol * 10.0
            
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
                    
        return s + penalty

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons and outer side
    # x, y in [-4, 4]
    # angle in [0, pi/3]
    # s in [2.0, 6.0]
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
        (2.0, 6.0)   # s
    ]
    
    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search
    try:
        res = differential_evolution(
            compute_objective_symmetric,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Fallback or Refinement: Try a few local minimizations from good symmetric starts
    # Start with a known good configuration: 1 center, 6 ring, 5 ring? 
    # Symmetric version: 
    # If we use 6-fold, we can't have a center hexagon unless it's at (0,0) and rotated?
    # A hexagon at (0,0) rotated by 0 is invariant under 60 deg rotation? No, it maps to itself.
    # So we can have one hexagon at center.
    # Then we have 11 left. 11 is not divisible by 6.
    # So pure 6-fold symmetry with 12 hexagons implies no center hexagon, or 6+6?
    # 12 = 6 * 2. So 2 hexagons per orbit.
    # One orbit could be at radius r1, angle 0.
    # The other at radius r2, angle 30?
    
    # Let's try a specific symmetric configuration manually to ensure we have a valid fallback
    # Configuration: 2 hexagons in each of 6 sectors.
    # Sector 0: Hex at (d1, 0) angle 0, Hex at (d2, 30deg) angle 30deg?
    
    # Let's just run a few L-BFGS-B minimizations from the DE result and some heuristics.
    
    if best_params is not None:
        # Polish with L-BFGS-B
        try:
            res_local = minimize(
                compute_objective_symmetric,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x
        except Exception:
            pass
    
    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple grid or known approximate
        # Try a few random symmetric starts
        for _ in range(5):
            p = np.array([
                np.random.uniform(1, 3), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                np.random.uniform(2, 4), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                4.0
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
                
    x1, y1, a1, x2, y2, a2, s = best_params
    
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
    outer_side = s
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
