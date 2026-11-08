import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    """
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    for phi_deg in np.linspace(0, 60, 600):
        phi = np.radians(phi_deg)
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        widths = max_projs - min_projs
        
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        if np.any(lo > hi + 1e-12):
            continue
        
        mid = (lo + hi) / 2
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    if best_center is None:
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _containment_penalty(inner_verts, outer_verts):
    """Penalty for inner hexagon vertices not being inside outer hexagon."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                penalty += (-cross) / edge_len
    return penalty

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Total penalty: containment + pairwise disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_vertices(cx, cy, 1.0, ang))
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth * 10.0
    
    return cont_pen + disj_pen

def _honeycomb_positions(count, pattern):
    """Generate positions for 'count' hexagons in honeycomb pattern."""
    sqrt3 = np.sqrt(3)
    scale = 1.001
    positions = []
    for j, row_len in enumerate(pattern):
        for i in range(row_len):
            x = (i + 0.5 * (j % 2)) * sqrt3 * scale
            y = j * 1.5 * scale
            positions.append((x, y))
    return positions[:count]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_result = None
    
    patterns = [
        [3, 4, 3, 2], [4, 3, 3, 2], [2, 3, 4, 3], [3, 3, 4, 2],
        [4, 4, 2, 2], [2, 2, 4, 4], [5, 3, 2, 2], [2, 2, 3, 5],
        [6, 3, 3], [3, 3, 6], [4, 4, 4], [2, 4, 4, 2],
        [4, 2, 4, 2], [2, 4, 2, 4], [3, 5, 4], [4, 5, 3],
        [5, 4, 3], [3, 4, 5], [2, 5, 5], [5, 5, 2], [5, 2, 5],
    ]
    
    for pattern in patterns:
        if sum(pattern) != 12 or len(pattern) > 6:
            continue
        
        if time.time() > deadline - 5.0:
            break
        
        positions = _honeycomb_positions(12, pattern)
        if len(positions) != 12:
            continue
        
        rot = 30.0
        inner = [(x, y, rot) for (x, y) in positions]
        inner_angles = [r[2] for r in inner]
        inner_sides = [1.0] * 12
        
        cx, cy, side, angle = _min_enclosing_hexagon(inner, inner_angles, inner_sides)
        side_safe = side * 1.0001 + 1e-6
        
        if side_safe < best_score:
            best_score = side_safe
            best_result = (inner, (cx, cy), side_safe, angle)
    
    # Optimization refinement: try to improve the best configuration found
    if best_result is not None and time.time() < deadline - 30.0:
        inner0, (cx0, cy0), side0, angle0 = best_result
        params0 = np.array([x for (x, y, a) in inner0 for x in [x, y, a]])
        
        for outer_angle in [angle0, angle0 + 15.0, angle0 - 15.0]:
            if time.time() > deadline - 10.0:
                break
            
            for outer_side in [side0 * 0.98, side0 * 0.96, side0 * 0.94]:
                if time.time() > deadline - 10.0:
                    break
                
                outer_center = (cx0, cy0)
                
                def objective(p):
                    return _total_penalty(p, outer_side, outer_center, outer_angle)
                
                try:
                    result = minimize(
                        objective, params0,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue
                
                if opt_val < 1e-6:
                    inner_opt = [(opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]) for i in range(N)]
                    # Verify with exact enclosing hexagon
                    inner_angles_opt = [r[2] for r in inner_opt]
                    cx_opt, cy_opt, side_opt, angle_opt = _min_enclosing_hexagon(inner_opt, inner_angles_opt, [1.0]*12)
                    side_opt_safe = side_opt * 1.0001 + 1e-6
                    
                    if side_opt_safe < best_score:
                        best_score = side_opt_safe
                        best_result = (inner_opt, (cx_opt, cy_opt), side_opt_safe, angle_opt)
    
    if best_result is None:
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
