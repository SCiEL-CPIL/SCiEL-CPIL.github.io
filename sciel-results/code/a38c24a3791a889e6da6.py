# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def _check_separation(hex1, hex2):
    """Check if two hexagons (arrays of 6 vertices each) are separated by SAT."""
    def get_normals(vertices):
        normals = np.zeros((6, 2))
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            normals[i] = n
        return normals

    n1 = get_normals(hex1)
    n2 = get_normals(hex2)
    all_normals = np.vstack([n1, n2])
    norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
    all_normals = all_normals / norms

    for n in all_normals:
        proj1 = hex1 @ n
        proj2 = hex2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _required_outer_size(all_vertices, outer_center, outer_angle_deg):
    """Compute the minimum side length of a regular hexagon with given center and angle
    that contains all given vertices."""
    a_rad = np.deg2rad(outer_angle_deg)
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each normal, the maximum projection of any vertex relative to center
    shifted = all_vertices - outer_center
    proj = shifted @ normals.T
    max_proj = np.max(proj, axis=0)
    
    # The apothem must be at least max_proj
    # apothem = S * sqrt(3)/2
    S = np.max(max_proj) / (np.sqrt(3) / 2)
    return S

def _validate(inner_hex_data, outer_hex_data, side):
    """Full validation check. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False

    inner_verts = []
    for i in range(11):
        cx, cy, ang = inner_hex_data[i]
        inner_verts.append(_hex_vertices(cx, cy, 1.0, ang))

    # Check separation using SAT
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j]):
                return False

    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oang)
    shifted_outer = outer_verts - np.array([ox, oy])
    shifted_inner = []
    for v in inner_verts:
        shifted_inner.append(v - np.array([ox, oy]))

    for iv in shifted_inner:
        for p in iv:
            for k in range(6):
                p1 = shifted_outer[k]
                p2 = shifted_outer[(k + 1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False

    for sv in shifted_inner:
        for k in range(6):
            d = np.linalg.norm(sv[(k+1)%6] - sv[k])
            if abs(d - 1.0) > 1e-8:
                return False

    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    h = np.sqrt(3)
    
    # Configuration 1: Standard Flower + 4 in gaps (compact)
    centers1 = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [h*np.cos(np.deg2rad(60)), h*np.sin(np.deg2rad(60))],
        [h*np.cos(np.deg2rad(120)), h*np.sin(np.deg2rad(120))],
        [h*np.cos(np.deg2rad(180)), h*np.sin(np.deg2rad(180))],
        [h*np.cos(np.deg2rad(240)), h*np.sin(np.deg2rad(240))],
        [h*np.cos(np.deg2rad(300)), h*np.sin(np.deg2rad(300))],
        [2*h*np.cos(np.deg2rad(30)), 2*h*np.sin(np.deg2rad(30))],
        [2*h*np.cos(np.deg2rad(150)), 2*h*np.sin(np.deg2rad(150))],
        [2*h*np.cos(np.deg2rad(210)), 2*h*np.sin(np.deg2rad(210))],
        [2*h*np.cos(np.deg2rad(330)), 2*h*np.sin(np.deg2rad(330))],
    ])
    
    # Configuration 2: 3-4-3 grid pattern
    sp = np.sqrt(3)  # hex spacing
    centers3 = np.array([
        [0, 0], [-sp, 0], [sp, 0],
        [-sp/2, h], [sp/2, h],
        [-sp, h], [0, h], [sp, h],
        [-sp/2, 2*h], [sp/2, 2*h],
        [0, 2*h]
    ])
    
    # Configuration 3: Compact diamond
    centers4 = np.array([
        [0, 0],
        [-h, 0], [h, 0],
        [-h/2, h*np.sqrt(3)/2], [h/2, h*np.sqrt(3)/2],
        [-h/2, -h*np.sqrt(3)/2], [h/2, -h*np.sqrt(3)/2],
        [-3*h/2, h*np.sqrt(3)/2], [3*h/2, h*np.sqrt(3)/2],
        [-3*h/2, -h*np.sqrt(3)/2], [3*h/2, -h*np.sqrt(3)/2],
    ])
    
    configs = [centers1, centers3, centers4]
    
    best_S = float('inf')
    best_inner = None
    best_outer = None
    
    for centers in configs:
        if time.time() > deadline:
            break
        
        # Get all vertices for this configuration
        all_verts = []
        for i in range(11):
            v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Check separation
        separated = True
        for i in range(11):
            for j in range(i + 1, 11):
                if not _check_separation(all_verts[i*6:(i+1)*6], all_verts[j*6:(j+1)*6]):
                    separated = False
                    break
            if not separated:
                break
        if not separated:
            continue
        
        # Optimize outer hexagon: center (cx, cy) and angle
        # Objective: minimize side length required to contain all vertices
        
        def obj_outer(params):
            cx, cy, angle = params[0], params[1], params[2]
            S = _required_outer_size(all_verts, np.array([cx, cy]), angle)
            return S
        
        # Multi-start optimization
        best_local_S = float('inf')
        best_local_params = None
        
        # Try several initial angles
        for init_angle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
            # Initial center: centroid of vertices
            cx0, cy0 = np.mean(all_verts[:, 0]), np.mean(all_verts[:, 1])
            
            x0 = np.array([cx0, cy0, float(init_angle)])
            
            try:
                res = minimize(obj_outer, x0, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_params = res.x.copy()
            except Exception:
                pass
        
        if best_local_params is not None and best_local_S < best_S:
            best_S = best_local_S
            best_inner = centers.copy()
            best_outer = best_local_params.copy()
    
    if best_inner is None:
        # Fallback
        best_inner = centers1
        centroid = np.mean(best_inner, axis=0)
        best_outer = np.array([centroid[0], centroid[1], 0.0])
        all_verts = np.vstack([_hex_vertices(c[0], c[1], 1.0, 0.0) for c in best_inner])
        best_S = _required_outer_size(all_verts, best_outer[:2], best_outer[2])
    
    # Local refinement: optimize positions of all 11 hexagons + outer hexagon
    if time.time() < deadline - 60:
        def _separation_penalty(centers):
            """Penalty for overlapping hexagons."""
            pen = 0.0
            n = len(centers)
            a_rad = 0.0
            normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            for i in range(n):
                for j in range(i + 1, n):
                    d = centers[j] - centers[i]
                    proj = normals @ d
                    min_gap = np.sqrt(3) - np.max(np.abs(proj))
                    if min_gap > 0:
                        pen += min_gap ** 2
            return pen
        
        def _full_objective(params):
            centers = params[:22].reshape(11, 2)
            ox, oy, oang = params[22], params[23], params[24]
            
            # Separation penalty
            sep_pen = _separation_penalty(centers)
            
            # Compute all vertices
            all_verts = []
            for i in range(11):
                v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            
            # Required outer side
            S = _required_outer_size(all_verts, np.array([ox, oy]), oang)
            
            # Containment penalty
            outer_verts = _hex_vertices(ox, oy, S, oang)
            shifted_outer = outer_verts - np.array([ox, oy])
            cont_pen = 0.0
            for i in range(11):
                for p in all_verts[i*6:(i+1)*6]:
                    ps = p - np.array([ox, oy])
                    for k in range(6):
                        p1 = shifted_outer[k]
                        p2 = shifted_outer[(k + 1) % 6]
                        cross_val = (p2[0] - p1[0]) * (ps[1] - p1[1]) - (p2[1] - p1[1]) * (ps[0] - p1[0])
                        if cross_val < 0:
                            cont_pen += 10.0
                            break
                    else:
                        continue
                    break
            
            return S + sep_pen + cont_pen
        
        x0 = np.concatenate([best_inner.flatten(), best_outer])
        
        try:
            res = minimize(_full_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            new_centers = res.x[:22].reshape(11, 2)
            new_outer = res.x[22:25]
            
            # Verify the refined solution
            inner_hex_data_test = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data_test[i] = [new_centers[i, 0], new_centers[i, 1], 0.0]
            
            if _validate(inner_hex_data_test, new_outer, new_outer[2]):
                if new_outer[2] < best_S:
                    best_inner = new_centers
                    best_outer = new_outer
                    best_S = new_outer[2]
        except Exception:
            pass
    
    # Final refinement on outer hexagon only
    if time.time() < deadline - 30:
        all_verts = np.vstack([_hex_vertices(c[0], c[1], 1.0, 0.0) for c in best_inner])
        
        def _obj_angle_final(angle_deg):
            cx, cy = best_outer[0], best_outer[1]
            return _required_outer_size(all_verts, np.array([cx, cy]), angle_deg)
        
        a0 = best_outer[2] % 60
        a_lo = max(0, a0 - 5)
        a_hi = min(60, a0 + 5)
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        c = a_hi - gr * (a_hi - a_lo)
        d = a_lo + gr * (a_hi - a_lo)
        fc = _obj_angle_final(c)
        fd = _obj_angle_final(d)
        for _ in range(60):
            if abs(a_hi - a_lo) < 1e-12:
                break
            if fc < fd:
                a_hi = d
                d = c
                fd = fc
                c = a_hi - gr * (a_hi - a_lo)
                fc = _obj_angle_final(c)
            else:
                a_lo = c
                c = d
                fc = fd
                d = a_lo + gr * (a_hi - a_lo)
                fd = _obj_angle_final(d)
        
        ang_ref = (a_lo + a_hi) / 2
        S_ref = _obj_angle_final(ang_ref)
        
        if S_ref < best_S:
            best_S = S_ref
            best_outer = np.array([best_outer[0], best_outer[1], ang_ref])
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
    
    outer_hex_data = best_outer
    
    if not _validate(inner_hex_data, outer_hex_data, best_S):
        # Bump side slightly
        best_S += 1e-4
        if not _validate(inner_hex_data, outer_hex_data, best_S):
            best_S += 1e-3
            if not _validate(inner_hex_data, outer_hex_data, best_S):
                # Last resort: use a known valid configuration
                best_inner = np.array([
                    [0, 0], [-2.5, 0], [2.5, 0],
                    [-1.25, 2.17], [1.25, 2.17],
                    [-1.25, -2.17], [1.25, -2.17],
                    [-3.75, 2.17], [3.75, 2.17],
                    [-3.75, -2.17], [3.75, -2.17]
                ])
                inner_hex_data = np.zeros((11, 3))
                for i in range(11):
                    inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
                centroid = np.mean(best_inner, axis=0)
                best_outer = np.array([centroid[0], centroid[1], 0.0])
                all_verts = np.vstack([_hex_vertices(c[0], c[1], 1.0, 0.0) for c in best_inner])
                best_S = _required_outer_size(all_verts, best_outer[:2], best_outer[2])
                best_S += 1e-3
    
    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END
