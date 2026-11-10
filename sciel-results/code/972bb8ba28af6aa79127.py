import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.radians(angle_deg)
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    n1 = _hex_edge_normals(hex1[2])
    n2 = _hex_edge_normals(hex2[2])
    
    for normals in [n1, n2]:
        for normal in normals:
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 + tol < min2 or max2 + tol < min1:
                return True
    
    return False

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _validate(inner_data, outer_data, s):
    """Validate the packing."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    return True

def _min_enclosing_hex_side(centers, inner_angles=None):
    """Find minimum side length s of a regular hexagon (centered at origin, angle 0) containing all inner hexagons."""
    outer_normals = _hex_edge_normals(0.0)
    
    if inner_angles is None:
        inner_angles = np.zeros(len(centers))
    
    max_required_s = 0.0
    for i, center in enumerate(centers):
        inner_angle = inner_angles[i]
        inner_verts = _hex_vertices(0, 0, 1.0, inner_angle)
        for n in outer_normals:
            support = np.max(inner_verts @ n)
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _honeycomb_positions():
    """Generate initial honeycomb positions for 12 hexagons."""
    sqrt3 = np.sqrt(3)
    col_spacing = 1.5
    row_spacing = sqrt3
    
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * col_spacing
            if col % 2 == 0:
                y = row * row_spacing
            else:
                y = (row + 0.5) * row_spacing
            positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return np.array(positions)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    sqrt3 = np.sqrt(3)
    
    # Initial honeycomb placement
    inner_data = _honeycomb_positions()
    centers = inner_data[:, :2]
    inner_angles = np.zeros(12)
    
    s = _min_enclosing_hex_side(centers, inner_angles)
    s *= 1.001
    outer_data = np.array([0.0, 0.0, 0.0])
    
    if not _validate(inner_data, outer_data, s):
        s = 6.0
        outer_data = np.array([0.0, 0.0, 0.0])
    
    best_value = 1.0 / s
    best_result = (inner_data.copy(), outer_data.copy(), s)
    
    # Precompute outer normals and inner vertex set (angle 0)
    outer_normals = _hex_edge_normals(0.0)
    inner_verts_base = _hex_vertices(0, 0, 1.0, 0.0)  # 6x2
    inner_normals = _hex_edge_normals(0.0)  # 6x2
    
    # Precompute support of inner hexagon in each outer normal direction
    support_in_outer_normals = np.array([np.max(inner_verts_base @ n) for n in outer_normals])
    
    # Precompute all vertex pairs for separation: for each pair of normals, 
    # we need the min/max projections of the base hexagon
    # For hexagon at position p: projection of vertex v_k onto normal n is (p + v_k) . n = p.n + v_k.n
    # So proj_min = p.n + min_k(v_k.n), proj_max = p.n + max_k(v_k.n)
    
    # Precompute for each normal the min and max of base hex vertices
    base_proj_min = np.array([np.min(inner_verts_base @ n) for n in inner_normals])
    base_proj_max = np.array([np.max(inner_verts_base @ n) for n in inner_normals])
    
    # Also for outer normals (same as inner since angle 0)
    outer_base_proj_min = np.array([np.min(inner_verts_base @ n) for n in outer_normals])
    outer_base_proj_max = np.array([np.max(inner_verts_base @ n) for n in outer_normals])
    
    # All 12 normals (6 from hex1 + 6 from hex2, but both are angle 0 so same set)
    # Actually for separation we use normals from both hexagons, but both have angle 0
    # so it's just the same 6 normals
    sep_normals = inner_normals  # 6x2
    
    # Precompute for each normal: min and max of base hex
    sep_base_min = np.array([np.min(inner_verts_base @ n) for n in sep_normals])
    sep_base_max = np.array([np.max(inner_verts_base @ n) for n in sep_normals])
    
    # All pairs
    all_pairs = [(i, j) for i in range(12) for j in range(i+1, 12)]
    n_pairs = len(all_pairs)
    
    def objective(params):
        inner_xy = params[:24].reshape(12, 2)
        s_val = params[24]
        
        if s_val < np.sqrt(12):
            return 1e6
        
        penalty = 0.0
        apothem = s_val * sqrt3 / 2
        
        # Containment: for each inner hex, check against each outer normal
        # max_proj = center . n + support_in_outer_normals[k]
        # Need max_proj <= apothem
        for i in range(12):
            center = inner_xy[i]
            for k in range(6):
                n = outer_normals[k]
                max_proj = center @ n + support_in_outer_normals[k]
                if max_proj > apothem + 1e-9:
                    penalty += (max_proj - apothem) ** 2
        
        # Separation: for each pair, check all 6 normals
        for idx in range(n_pairs):
            i, j = all_pairs[idx]
            ci = inner_xy[i]
            cj = inner_xy[j]
            
            separated = False
            min_gap = np.inf
            for k in range(6):
                n = sep_normals[k]
                # Hex i projection: [ci.n + sep_base_min[k], ci.n + sep_base_max[k]]
                # Hex j projection: [cj.n + sep_base_min[k], cj.n + sep_base_max[k]]
                ci_n = ci @ n
                cj_n = cj @ n
                
                min1 = ci_n + sep_base_min[k]
                max1 = ci_n + sep_base_max[k]
                min2 = cj_n + sep_base_min[k]
                max2 = cj_n + sep_base_max[k]
                
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
                
                gap = min(min2 - max1, min1 - max2)
                if gap < min_gap:
                    min_gap = gap
            
            if not separated and min_gap < 0:
                penalty += min_gap ** 2
        
        return s_val + 1000 * penalty
    
    # Phase 1: Affine transform optimization (4-parameter: rotation, sx, sy, s)
    base_positions = _honeycomb_positions()[:, :2]
    
    def affine_objective(params):
        angle = params[0]
        sx = params[1]
        sy = params[2]
        s_val = params[3]
        
        if s_val < np.sqrt(12):
            return 1e6
        if sx <= 0 or sy <= 0:
            return 1e6
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        transformed = np.zeros((12, 2))
        for i in range(12):
            x, y = base_positions[i]
            xs = x * sx
            ys = y * sy
            transformed[i, 0] = xs * cos_a - ys * sin_a
            transformed[i, 1] = xs * sin_a + ys * cos_a
        
        full_params = np.zeros(25)
        full_params[:24] = transformed.flatten()
        full_params[24] = s_val
        
        return objective(full_params)
    
    # Run differential evolution for affine params
    try:
        if time.time() < deadline - 120:
            affine_bounds = [(-np.pi, np.pi), (0.3, 2.0), (0.3, 2.0), (np.sqrt(12), 10.0)]
            affine_result = differential_evolution(
                affine_objective,
                affine_bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                polish=False,
                workers=-1
            )
            
            if affine_result.fun < 100:
                angle_opt = affine_result.x[0]
                sx_opt = affine_result.x[1]
                sy_opt = affine_result.x[2]
                
                cos_a = np.cos(angle_opt)
                sin_a = np.sin(angle_opt)
                
                transformed = np.zeros((12, 2))
                for i in range(12):
                    x, y = base_positions[i]
                    xs = x * sx_opt
                    ys = y * sy_opt
                    transformed[i, 0] = xs * cos_a - ys * sin_a
                    transformed[i, 1] = xs * sin_a + ys * cos_a
                
                s_exact = _min_enclosing_hex_side(transformed, inner_angles)
                s_opt = s_exact * 1.0001
                
                inner_opt = np.zeros((12, 3))
                inner_opt[:, 0] = transformed[:, 0]
                inner_opt[:, 1] = transformed[:, 1]
                inner_opt[:, 2] = 0.0
                outer_opt = np.array([0.0, 0.0, 0.0])
                
                if _validate(inner_opt, outer_opt, s_opt):
                    value = 1.0 / s_opt
                    if value > best_value:
                        best_value = value
                        best_result = (inner_opt, outer_opt, s_opt)
    except Exception:
        pass
    
    # Phase 2: Full 24-DOF Nelder-Mead optimization with multiple restarts
    init_params = np.zeros(25)
    init_params[:24] = best_result[0][:, :2].flatten()
    init_params[24] = best_result[2]
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params0 = init_params.copy()
        else:
            perturbation = 0.2 * np.random.randn(24)
            params0 = init_params.copy()
            params0[:24] += perturbation
            params0[24] = best_result[2] + 0.2 * np.random.randn()
        
        try:
            result = minimize(
                objective,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < 100:
                final_params = result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
                            # Update init for next restart
                            init_params[:24] = inner_xy_final.flatten()
                            init_params[24] = s_final
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 40:
        current_inner = best_result[0].copy()
        current_s = best_result[2]
        
        polish_params = np.zeros(25)
        polish_params[:24] = current_inner[:, :2].flatten()
        polish_params[24] = current_s
        
        try:
            polish_result = minimize(
                objective,
                polish_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 100:
                final_params = polish_result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    # Phase 4: Final ultra-precise polish if time permits
    if time.time() < deadline - 20:
        current_inner = best_result[0].copy()
        current_s = best_result[2]
        
        polish_params = np.zeros(25)
        polish_params[:24] = current_inner[:, :2].flatten()
        polish_params[24] = current_s
        
        try:
            polish_result = minimize(
                objective,
                polish_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 100:
                final_params = polish_result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    return best_result
