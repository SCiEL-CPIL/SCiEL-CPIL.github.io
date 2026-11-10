import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(vertices):
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm > 1e-12:
            normals[i] = [nx / norm, ny / norm]
    return normals

def _min_separation_gap(v1, n1, v2, n2):
    all_normals = np.vstack([n1, n2])
    proj1 = v1 @ all_normals.T
    proj2 = v2 @ all_normals.T
    min1 = proj1.min(axis=0)
    max1 = proj1.max(axis=0)
    min2 = proj2.min(axis=0)
    max2 = proj2.max(axis=0)
    gaps = np.maximum(min2 - max1, min1 - max2)
    return np.min(gaps)

def _all_pairs_separated(inner_hexes):
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        nm = _edge_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            gap = _min_separation_gap(vertices_list[i], normals_list[i], vertices_list[j], normals_list[j])
            if gap <= 1e-6:
                return False
    return True

def _outer_contains_inner(outer_center, outer_side, outer_angle, inner_hexes, tol=1e-6):
    outer_v = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(len(inner_hexes)):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -tol:
                    return False
    return True

def _compute_side_length(inner_hexes, outer_center, outer_angle):
    outer_v_template = _hex_vertices(0, 0, 1.0, outer_angle)
    outer_normals = _edge_normals(outer_v_template)
    max_s = 0.0
    for i in range(len(inner_hexes)):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        dp = v - outer_center
        projs = dp @ outer_normals.T
        max_proj = np.max(projs)
        required_s = max_proj * 2.0 / np.sqrt(3.0)
        if required_s > max_s:
            max_s = required_s
    return max_s

def _objective(params, outer_angle):
    inner_hexes = params.reshape(11, 3)
    side = _compute_side_length(inner_hexes, np.array([0.0, 0.0]), outer_angle)
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        nm = _edge_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = _min_separation_gap(vertices_list[i], normals_list[i], vertices_list[j], normals_list[j])
            if gap < 0.0:
                penalty += (gap * gap) * 100.0
    return side + penalty

def _build_initial_config():
    sqrt3 = np.sqrt(3)
    d1 = sqrt3
    d2 = 2 * sqrt3
    hexes = np.array([[0.0, 0.0, 0.0]])
    for i in range(6):
        angle = i * 60
        x = d1 * np.cos(np.radians(angle))
        y = d1 * np.sin(np.radians(angle))
        hexes = np.vstack([hexes, [x, y, 0.0]])
    for angle in [0, 90, 180, 270]:
        x = d2 * np.cos(np.radians(angle))
        y = d2 * np.sin(np.radians(angle))
        hexes = np.vstack([hexes, [x, y, 0.0]])
    return hexes

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 610.0
    outer_center = np.array([0.0, 0.0])
    best_params = None
    best_side = float('inf')
    best_angle = 0.0
    init = _build_initial_config()
    
    for angle in [0, 30]:
        if time.time() > deadline - 60:
            break
        side = _compute_side_length(init, outer_center, angle)
        if _all_pairs_separated(init):
            if side < best_side:
                best_params = init.copy()
                best_side = side
                best_angle = angle
                
    if best_params is None:
        best_params = init
        best_angle = 0.0
        best_side = _compute_side_length(best_params, outer_center, best_angle)
        
    # Optimize positions only first
    def obj_pos_only(pos_params):
        inner_hexes = np.zeros((11, 3))
        inner_hexes[:, 0] = pos_params[0::2]
        inner_hexes[:, 1] = pos_params[1::2]
        inner_hexes[:, 2] = 0.0
        return _objective(inner_hexes.flatten(), best_angle)
        
    if time.time() < deadline - 120:
        try:
            initial_pos = best_params[:, :2].flatten()
            nm_result = minimize(
                obj_pos_only,
                initial_pos,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result.fun < best_side:
                candidate_pos = nm_result.x
                candidate = np.zeros((11, 3))
                candidate[:, 0] = candidate_pos[0::2]
                candidate[:, 1] = candidate_pos[1::2]
                candidate[:, 2] = 0.0
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        if side < best_side:
                            best_params = candidate
                            best_side = side
        except Exception:
            pass
            
    # Optimize full parameters (including angles)
    if time.time() < deadline - 60:
        try:
            nm_result2 = minimize(
                _objective,
                best_params.flatten(),
                args=(best_angle,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result2.fun < best_side:
                candidate = nm_result2.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        if side < best_side:
                            best_params = candidate
                            best_side = side
        except Exception:
            pass
            
    final_side = _compute_side_length(best_params, outer_center, best_angle)
    if not _all_pairs_separated(best_params) or not _outer_contains_inner(outer_center, final_side, best_angle, best_params):
        best_params = init
        best_angle = 0.0
        final_side = _compute_side_length(best_params, outer_center, best_angle)
        
    return best_params, np.array([0.0, 0.0, float(best_angle)]), float(final_side)
