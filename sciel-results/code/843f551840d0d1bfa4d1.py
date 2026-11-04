import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_composite_hull(centers, angles):
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        verts = _hex_vertices(cx, cy, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
        return hull_verts
    except Exception:
        return all_verts

def _hexagon_distance_field_fast(cx1, cy1, a1, cx2, cy2, a2):
    """
    Fast computation of minimum distance between two unit hexagons.
    Positive if disjoint, negative if overlapping.
    """
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    
    min_dist = float('inf')
    
    # Vertex-to-edge: v1 vertices vs v2 edges
    for vi in v1:
        for j in range(6):
            vj = v2[j]
            vj1 = v2[(j+1)%6]
            edge = vj1 - vj
            d2 = np.dot(edge, edge)
            if d2 < 1e-12:
                dist2 = np.dot(vi - vj, vi - vj)
            else:
                t = np.clip(np.dot(vi - vj, edge) / d2, 0, 1)
                closest = vj + t * edge
                diff = vi - closest
                dist2 = np.dot(diff, diff)
            if dist2 < min_dist:
                min_dist = dist2
    
    # Vertex-to-edge: v2 vertices vs v1 edges
    for vi in v2:
        for j in range(6):
            vj = v1[j]
            vj1 = v1[(j+1)%6]
            edge = vj1 - vj
            d2 = np.dot(edge, edge)
            if d2 < 1e-12:
                dist2 = np.dot(vi - vj, vi - vj)
            else:
                t = np.clip(np.dot(vi - vj, edge) / d2, 0, 1)
                closest = vj + t * edge
                diff = vi - closest
                dist2 = np.dot(diff, diff)
            if dist2 < min_dist:
                min_dist = dist2
    
    # Edge-edge distances
    for i1 in range(6):
        p1 = v1[i1]
        p2 = v1[(i1+1)%6]
        for i2 in range(6):
            q1 = v2[i2]
            q2 = v2[(i2+1)%6]
            r = p2 - p1
            s = q2 - q1
            r1 = p1 - q1
            a = np.dot(r, r)
            e = np.dot(s, s)
            f = np.dot(s, r1)
            
            if a < 1e-12 and e < 1e-12:
                dist2 = np.dot(r1, r1)
            elif a < 1e-12:
                t = np.clip(f / e, 0, 1)
                diff = p1 - (q1 + t * s)
                dist2 = np.dot(diff, diff)
            elif e < 1e-12:
                t = np.clip(-np.dot(r, r1) / a, 0, 1)
                diff = p1 + t * r - q1
                dist2 = np.dot(diff, diff)
            else:
                c = np.dot(r, r1)
                d = np.dot(r, s)
                denom = a * e - d * d
                if denom < 1e-12:
                    t = 0.0
                    u = np.clip(f / e, 0, 1)
                else:
                    t = np.clip((d * f - c * e) / denom, 0, 1)
                    u = np.clip((a * f - d * c) / denom, 0, 1)
                closest1 = p1 + t * r
                closest2 = q1 + u * s
                diff = closest1 - closest2
                dist2 = np.dot(diff, diff)
            if dist2 < min_dist:
                min_dist = dist2
    
    return np.sqrt(min_dist) if min_dist > 0 else 0.0

def _overlap_penalty_fast(centers, angles, beta=80.0):
    """
    Fast smooth penalty for overlapping hexagons.
    """
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = _hexagon_distance_field_fast(
                centers[i][0], centers[i][1], angles[i],
                centers[j][0], centers[j][1], angles[j]
            )
            if dist < 0.05:
                penalty += np.exp(-beta * dist) - 1.0
    return penalty

def _objective_joint(vars, angles, n_inner):
    """
    Joint objective: optimize inner positions AND outer parameters together.
    vars: [cx1, cy1, ..., cxN, cyN, outer_cx, outer_cy, outer_angle_deg]
    """
    centers = np.array([[vars[2*i], vars[2*i+1]] for i in range(n_inner)])
    outer_center = np.array([vars[2*n_inner], vars[2*n_inner+1]])
    outer_angle = vars[2*n_inner+2]
    
    composite = _compute_composite_hull(centers, angles)
    S = _min_enclosing_hex_side(composite, outer_angle, outer_center)
    
    pen = _overlap_penalty_fast(centers, angles)
    
    return S + 200.0 * pen

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
    
    # Outer ring: 4 hexagons at distance 3 from center
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
        
    centers = np.array(centers_list)
    angles = np.array(angles_list)
    n_inner = len(centers)
    
    best_S = float('inf')
    best_centers = centers.copy()
    best_outer_angle = 0.0
    best_outer_center = np.array([0.0, 0.0])
    
    # Phase 1: Quick inner optimization with outer at origin, angle 0
    def obj_inner_only(vars):
        c = np.array([[vars[2*i], vars[2*i+1]] for i in range(n_inner)])
        composite = _compute_composite_hull(c, angles)
        S = _min_enclosing_hex_side(composite, 0.0, np.array([0.0, 0.0]))
        pen = _overlap_penalty_fast(c, angles)
        return S + 100.0 * pen
    
    x0_inner = np.zeros(2 * n_inner)
    for i in range(n_inner):
        x0_inner[2*i] = centers[i][0]
        x0_inner[2*i + 1] = centers[i][1]
    
    try:
        if time.time() < DEADLINE - 60.0:
            res = minimize(obj_inner_only, x0_inner, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            new_centers = np.array([[res.x[2*i], res.x[2*i+1]] for i in range(n_inner)])
            pen = _overlap_penalty_fast(new_centers, angles)
            if pen < 1e-4:
                composite = _compute_composite_hull(new_centers, angles)
                S_trial = _min_enclosing_hex_side(composite, 0.0, np.array([0.0, 0.0]))
                if S_trial < best_S:
                    best_S = S_trial
                    best_centers = new_centers
    except Exception:
        pass
    
    # Phase 2: Joint optimization of inner positions + outer center + outer angle
    # vars: [cx1, cy1, ..., cxN, cyN, outer_cx, outer_cy, outer_angle]
    x0_joint = np.zeros(2 * n_inner + 3)
    for i in range(n_inner):
        x0_joint[2*i] = best_centers[i][0]
        x0_joint[2*i + 1] = best_centers[i][1]
    x0_joint[2*n_inner] = 0.0
    x0_joint[2*n_inner+1] = 0.0
    x0_joint[2*n_inner+2] = 0.0
    
    try:
        if time.time() < DEADLINE - 30.0:
            res = minimize(_objective_joint, x0_joint, args=(angles, n_inner),
                           method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_S + 1.0:  # Allow some penalty in objective
                new_centers = np.array([[res.x[2*i], res.x[2*i+1]] for i in range(n_inner)])
                new_outer_center = np.array([res.x[2*n_inner], res.x[2*n_inner+1]])
                new_outer_angle = res.x[2*n_inner+2]
                
                pen = _overlap_penalty_fast(new_centers, angles)
                if pen < 1e-4:
                    composite = _compute_composite_hull(new_centers, angles)
                    S_trial = _min_enclosing_hex_side(composite, new_outer_angle, new_outer_center)
                    if S_trial < best_S:
                        best_S = S_trial
                        best_centers = new_centers
                        best_outer_center = new_outer_center
                        best_outer_angle = new_outer_angle
    except Exception:
        pass
    
    # Phase 3: Fine-tune outer parameters only (inner fixed)
    def outer_obj(vars):
        ocx, ocy, oangle = vars
        composite = _compute_composite_hull(best_centers, angles)
        return _min_enclosing_hex_side(composite, oangle, np.array([ocx, ocy]))
    
    x0_outer = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    try:
        if time.time() < DEADLINE - 10.0:
            res = minimize(outer_obj, x0_outer, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res.fun < best_S:
                best_S = res.fun
                best_outer_center = res.x[:2]
                best_outer_angle = res.x[2]
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish
    # Recompute exact side length and do one more tight optimization
    composite = _compute_composite_hull(best_centers, angles)
    final_S = _min_enclosing_hex_side(composite, best_outer_angle, best_outer_center)
    if final_S < best_S:
        best_S = final_S
    
    # One more very tight polish on outer parameters
    try:
        if time.time() < DEADLINE - 2.0:
            res = minimize(outer_obj, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]),
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11})
            if res.fun < best_S:
                best_S = res.fun
                best_outer_center = res.x[:2]
                best_outer_angle = res.x[2]
    except Exception:
        pass
    
    # Final recomputation
    composite = _compute_composite_hull(best_centers, angles)
    final_S = _min_enclosing_hex_side(composite, best_outer_angle, best_outer_center)
    if final_S < best_S:
        best_S = final_S
    
    # Build output
    inner = []
    for i in range(n_inner):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])
    
    outer_center_out = (best_outer_center[0], best_outer_center[1])
    
    return inner, outer_center_out, best_S, best_outer_angle
