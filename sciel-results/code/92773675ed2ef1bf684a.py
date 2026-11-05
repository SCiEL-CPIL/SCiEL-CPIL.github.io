import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    thetas = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    thetas = angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _min_enclosing_hex_side_fast(centers, angles, outer_angle_deg, outer_center):
    """Fast computation using precomputed vertex offsets."""
    n = len(centers)
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        a = np.deg2rad(angles[i])
        thetas = a + np.arange(6) * np.pi / 3
        all_verts[i*6:(i+1)*6, 0] = centers[i][0] + np.cos(thetas)
        all_verts[i*6:(i+1)*6, 1] = centers[i][1] + np.sin(thetas)
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = all_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_composite_hull(centers, angles):
    n = len(centers)
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        a = np.deg2rad(angles[i])
        thetas = a + np.arange(6) * np.pi / 3
        all_verts[i*6:(i+1)*6, 0] = centers[i][0] + np.cos(thetas)
        all_verts[i*6:(i+1)*6, 1] = centers[i][1] + np.sin(thetas)
    try:
        hull = ConvexHull(all_verts)
        return all_verts[hull.vertices]
    except Exception:
        return all_verts

def _min_enclosing_hex_side_smooth(composite_verts, outer_angle_deg, outer_center, beta=100.0):
    """Smooth approximation of min enclosing hex side using soft-max."""
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    abs_dots = np.abs(dots)
    # Soft max over all projections
    max_val = np.logaddexp.reduce(abs_dots.ravel()) / 1.0  # This isn't quite right
    # Better: use log-sum-exp for soft max
    flat = abs_dots.ravel()
    m = np.max(flat)
    soft_max = m + np.log(np.sum(np.exp(flat - m)))
    return 2 * soft_max / np.sqrt(3)

def _hexagon_distance_field(cx1, cy1, a1, cx2, cy2, a2):
    """
    Compute minimum distance between two unit hexagons (positive if disjoint, negative if overlapping).
    Vectorized vertex-to-edge checks.
    """
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    
    min_dist = float('inf')
    
    # Vectorized: each vertex of v1 against each edge of v2
    for vi in v1:
        # Edges of v2: (v2[j], v2[j+1]) for j in range(6)
        p0 = v2
        p1 = np.roll(v2, -1, axis=0)
        r = p1 - p0
        d2 = np.sum(r * r, axis=1)
        # Avoid division by zero
        d2_safe = np.where(d2 < 1e-24, 1.0, d2)
        t = np.sum((vi - p0) * r, axis=1) / d2_safe
        t = np.clip(t, 0, 1)
        closest = p0 + t[:, None] * r
        dists = np.sqrt(np.sum((vi - closest) ** 2, axis=1))
        min_dist = min(min_dist, np.min(dists))
    
    # Vectorized: each vertex of v2 against each edge of v1
    for vi in v2:
        p0 = v1
        p1 = np.roll(v1, -1, axis=0)
        r = p1 - p0
        d2 = np.sum(r * r, axis=1)
        d2_safe = np.where(d2 < 1e-24, 1.0, d2)
        t = np.sum((vi - p0) * r, axis=1) / d2_safe
        t = np.clip(t, 0, 1)
        closest = p0 + t[:, None] * r
        dists = np.sqrt(np.sum((vi - closest) ** 2, axis=1))
        min_dist = min(min_dist, np.min(dists))
    
    return min_dist

def _overlap_penalty(centers, angles, beta=50.0):
    """
    Smooth penalty for overlapping hexagons using distance field.
    Returns 0 if all disjoint, positive penalty if overlapping.
    """
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = _hexagon_distance_field(
                centers[i][0], centers[i][1], angles[i],
                centers[j][0], centers[j][1], angles[j]
            )
            if dist < -1e-10:
                # Overlapping - use smooth penalty
                penalty += np.exp(-beta * dist) - 1.0
    return penalty

def _joint_objective(vars, angles):
    """
    Joint objective: minimize outer hex side + overlap penalty.
    vars: [outer_cx, outer_cy, outer_angle, cx1, cy1, cx2, cy2, ..., cx11, cy11]
    """
    n_inner = len(angles)
    outer_cx = vars[0]
    outer_cy = vars[1]
    outer_angle = vars[2]
    
    # Extract inner centers
    centers = np.array([[vars[3 + 2*i], vars[4 + 2*i]] for i in range(n_inner)])
    
    # Compute enclosing side (fast path)
    S = _min_enclosing_hex_side_fast(centers, angles, outer_angle, np.array([outer_cx, outer_cy]))
    
    # Compute overlap penalty
    pen = _overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def _objective(vars, centers, angles):
    outer_cx, outer_cy, outer_angle = vars
    outer_center = np.array([outer_cx, outer_cy])
    S = _min_enclosing_hex_side_fast(centers, angles, outer_angle, outer_center)
    return S

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    eps = 1e-9
    centers_list = []
    angles_list = []
    
    # Center hexagon
    centers_list.append([0.0, 0.0])
    angles_list.append(30.0)
    
    # Inner ring: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
    
    # Outer ring: 4 hexagons at distance 3 from center, at 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
        
    centers = np.array(centers_list)
    angles = np.array(angles_list)
    n_inner = len(centers)
    
    # Initial outer hexagon: centered at origin, angle 30
    x0_outer = np.array([0.0, 0.0, 30.0])
    
    best_S = None
    best_vars_outer = x0_outer
    best_centers = centers.copy()
    
    # Phase 1: Optimize outer hexagon with fixed inner positions (Nelder-Mead)
    try:
        res = minimize(_objective, x0_outer, args=(centers, angles), method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if time.time() < DEADLINE - 30.0:
            if res.fun < (best_S if best_S is not None else float('inf')):
                best_S = res.fun
                best_vars_outer = res.x
    except Exception:
        pass
    
    # Phase 2: Joint optimization using L-BFGS-B
    # Build initial variable vector: [outer_cx, outer_cy, outer_angle, cx1, cy1, ..., cx11, cy11]
    x0_joint = np.zeros(3 + 2 * n_inner)
    x0_joint[0] = best_vars_outer[0]
    x0_joint[1] = best_vars_outer[1]
    x0_joint[2] = best_vars_outer[2]
    for i in range(n_inner):
        x0_joint[3 + 2*i] = centers[i][0]
        x0_joint[4 + 2*i] = centers[i][1]
    
    # Try joint optimization with different outer angle initializations
    joint_best_S = best_S if best_S is not None else float('inf')
    joint_best_vars = x0_joint.copy()
    
    for init_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
        if time.time() > DEADLINE - 25.0:
            break
        
        x0_trial = x0_joint.copy()
        x0_trial[2] = init_angle
        
        try:
            res = minimize(_joint_objective, x0_trial, args=(angles,), 
                          method='L-BFGS-B', 
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() < DEADLINE - 20.0:
                # Extract the actual S (without penalty) from the result
                outer_cx, outer_cy, outer_angle = res.x[0], res.x[1], res.x[2]
                new_centers = np.array([[res.x[3+2*i], res.x[4+2*i]] for i in range(n_inner)])
                S_trial = _min_enclosing_hex_side_fast(new_centers, angles, outer_angle, np.array([outer_cx, outer_cy]))
                pen_trial = _overlap_penalty(new_centers, angles)
                
                # Only accept if no significant overlap
                if pen_trial < 1e-8 and S_trial < joint_best_S:
                    joint_best_S = S_trial
                    joint_best_vars = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Polish with Nelder-Mead on outer hexagon with optimized inner positions
    if joint_best_S < (best_S if best_S is not None else float('inf')):
        best_S = joint_best_S
        best_vars_outer = joint_best_vars[:3].copy()
        best_centers = np.array([[joint_best_vars[3+2*i], joint_best_vars[4+2*i]] for i in range(n_inner)])
    
    # Phase 4: High-precision polish - fine-tune inner positions and outer hex
    if time.time() < DEADLINE - 10.0:
        # Try small perturbations of inner positions to see if we can squeeze more
        for n_inner in [len(angles)]:
            x0_fine = np.zeros(3 + 2 * n_inner)
            x0_fine[0] = best_vars_outer[0]
            x0_fine[1] = best_vars_outer[1]
            x0_fine[2] = best_vars_outer[2]
            for i in range(n_inner):
                x0_fine[3 + 2*i] = best_centers[i][0]
                x0_fine[4 + 2*i] = best_centers[i][1]
            
            try:
                res = minimize(_joint_objective, x0_fine, args=(angles,), 
                              method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if time.time() < DEADLINE - 5.0:
                    outer_cx, outer_cy, outer_angle = res.x[0], res.x[1], res.x[2]
                    new_centers = np.array([[res.x[3+2*i], res.x[4+2*i]] for i in range(n_inner)])
                    S_trial = _min_enclosing_hex_side_fast(new_centers, angles, outer_angle, np.array([outer_cx, outer_cy]))
                    pen_trial = _overlap_penalty(new_centers, angles)
                    
                    if pen_trial < 1e-10 and S_trial < best_S:
                        best_S = S_trial
                        best_vars_outer = np.array([outer_cx, outer_cy, outer_angle])
                        best_centers = new_centers
            except Exception:
                pass
    
    # Final polish: Nelder-Mead on outer hexagon with best inner positions
    if time.time() < DEADLINE - 3.0:
        x0_polish = best_vars_outer.copy()
        try:
            res = minimize(_objective, x0_polish, args=(best_centers, angles), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_vars_outer = res.x
        except Exception:
            pass
    
    # Final result
    outer_cx, outer_cy, outer_angle = best_vars_outer
    outer_center = (outer_cx, outer_cy)
    
    inner = []
    for i in range(len(angles)):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])
    
    # Recompute side precisely
    side = _min_enclosing_hex_side_fast(best_centers, angles, outer_angle, np.array([outer_cx, outer_cy]))
    
    return inner, outer_center, side, outer_angle
