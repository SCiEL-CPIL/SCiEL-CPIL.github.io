import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    """
    ov_cx, ov_cy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    
    # The 6 outward normals to the sides of the outer hexagon
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each inner hexagon, find the maximum projection onto each normal
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        hv_rel = hv - np.array([ov_cx, ov_cy])
        projections = hv_rel @ normals.T
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    return s_min


def _check_separation(inner):
    """
    Check if all pairs of inner hexagons are separated (non-overlapping) using SAT.
    Returns True if all pairs are separated.
    """
    n = len(inner)
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > 1e-6 or gap2 > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"
    
    def _separated(v1, v2):
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True
        
        return False
    
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"
    
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, "inner vertex outside outer hexagon"
    
    return True, ""


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    sqrt3 = np.sqrt(3.0)
    
    # Symmetric 12-hexagon arrangement:
    inner = np.array([
        [0.0, 0.0, 0.0],           # Center
        [3.0, 0.0, 0.0],           # Right
        [-3.0, 0.0, 0.0],          # Left
        [1.5, 2.5980762, 0.0],     # Upper right
        [-1.5, 2.5980762, 0.0],    # Upper left
        [1.5, -2.5980762, 0.0],    # Lower right
        [-1.5, -2.5980762, 0.0],   # Lower left
        [4.5, 2.5980762, 0.0],     # Far upper right
        [-4.5, 2.5980762, 0.0],    # Far upper left
        [4.5, -2.5980762, 0.0],    # Far lower right
        [-4.5, -2.5980762, 0.0],   # Far lower left
        [0.0, 5.1961524, 0.0],     # Top
    ])
    
    outer = np.array([0.0, 0.0, 0.0])
    
    s = _min_containing_hex_side(inner, np.array([0.0, 0.0]), 0.0)
    s += 0.01
    
    valid, msg = _check_validity(inner, outer, s)
    if not valid:
        print(f"Initial guess invalid: {msg}")
        s += 1.0
        valid, msg = _check_validity(inner, outer, s)
        if not valid:
            s = 20.0
            return inner, outer, s
    
    best_s = s
    best_inner = inner.copy()
    best_outer = outer.copy()
    print(f"Initial: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    try:
        def objective(params):
            """
            params: 38 values: 12*(x, y, angle_deg) + 2*(ocx, ocy)
            Returns: computed s + penalty for overlaps
            """
            inner_p = params[:36].reshape(12, 3)
            ocx = params[36]
            ocy = params[37]
            outer_center = np.array([ocx, ocy])
            
            s_val = _min_containing_hex_side(inner_p, outer_center, 0.0)
            
            penalty = 0.0
            inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner_p]
            
            for i in range(12):
                for j in range(i + 1, 12):
                    v1 = inner_verts[i]
                    v2 = inner_verts[j]
                    
                    edges = []
                    for k in range(6):
                        edges.append(v1[(k + 1) % 6] - v1[k])
                        edges.append(v2[(k + 1) % 6] - v2[k])
                    
                    separated = False
                    for edge in edges:
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-12:
                            continue
                        normal /= norm
                        
                        proj1 = v1 @ normal
                        proj2 = v2 @ normal
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        gap1 = min2 - max1
                        gap2 = min1 - max2
                        if gap1 > 1e-6 or gap2 > 1e-6:
                            separated = True
                            break
                    
                    if not separated:
                        d = np.linalg.norm(inner_p[i, :2] - inner_p[j, :2])
                        if d < 3.0:
                            penalty += (3.0 - d) ** 2 * 100.0
            
            return s_val + penalty
        
        x0 = np.concatenate([inner.flatten(), [0.0, 0.0]])
        
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-7,
                                   'adaptive': True})
        
        if result.fun < best_s:
            inner_opt = result.x[:36].reshape(12, 3)
            ocx_opt = result.x[36]
            ocy_opt = result.x[37]
            outer_center_opt = np.array([ocx_opt, ocy_opt])
            s_opt = _min_containing_hex_side(inner_opt, outer_center_opt, 0.0) + 0.001
            outer_opt = np.array([ocx_opt, ocy_opt, 0.0])
            
            valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
            if valid:
                best_inner = inner_opt
                best_outer = outer_opt
                best_s = s_opt
                print(f"Optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
            else:
                print(f"Optimized result invalid: {msg}")
        
        if time.time() > deadline:
            return best_inner, best_outer, best_s
        
        # Second round of optimization with tighter tolerances
        if time.time() < deadline - 120:
            result2 = minimize(objective, result.x, method='Nelder-Mead',
                              options={'maxiter': 15000, 'xatol': 1e-8, 'fatol': 1e-8,
                                       'adaptive': True})
            
            if result2.fun < best_s:
                inner_opt = result2.x[:36].reshape(12, 3)
                ocx_opt = result2.x[36]
                ocy_opt = result2.x[37]
                outer_center_opt = np.array([ocx_opt, ocy_opt])
                s_opt = _min_containing_hex_side(inner_opt, outer_center_opt, 0.0) + 0.001
                outer_opt = np.array([ocx_opt, ocy_opt, 0.0])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
                if valid:
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt
                    print(f"Further optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, using fallback")
        s = 20.0
        return best_inner, best_outer, s
    
    return best_inner, best_outer, best_s
