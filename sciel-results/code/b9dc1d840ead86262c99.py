import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

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
    
    overlaps = np.minimum(max1, max2) - np.maximum(min1, max2)
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

def compute_min_side(hexes):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
    that contains all given hexagons.
    """
    # The outer hexagon has fixed orientation (angle 0).
    # Its supporting lines are defined by the 6 outward normals of a hexagon with side 1, angle 0.
    # Normal angles: 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for h in hexes:
        proj = h @ normals.T
        max_proj = max(max_proj, np.max(proj))
        
    # For a regular hexagon of side s, the distance from center to side is s * sqrt(3) / 2.
    # So max_proj = s * sqrt(3) / 2 => s = max_proj * 2 / sqrt(3)
    s = max_proj * 2.0 / np.sqrt(3.0)
    return s

def construct():
    t0 = time.time()
    
    def check_time():
        return time.time() > DEADLINE

    def compute_objective_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns min_side + penalty.
        """
        if check_time():
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
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
        
        # Calculate minimal outer side
        s_min = compute_min_side(hexes)
        
        penalty = 0.0
        
        # Check overlaps
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
                    
        return s_min + penalty

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons
    # x, y in [-4, 4]
    # angle in [0, pi/3]
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
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
                np.random.uniform(2, 4), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3)
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
            if time.time() > DEADLINE - 5:
                break
                
    x1, y1, a1, x2, y2, a2 = best_params
    
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
        
        inner.append([float(x1_rot), float(y1_rot), float(np.degrees(a1_rot))])
        inner.append([float(x2_rot), float(y2_rot), float(np.degrees(a2_rot))])
        
    # Calculate final outer side length
    s_final = compute_min_side([hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner])
    
    return inner, (0.0, 0.0), float(s_final), 0.0

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
