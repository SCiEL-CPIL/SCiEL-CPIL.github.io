# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600

    def hex_vertices(cx, cy, d, angle_deg):
        a = np.radians(angle_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

    def separated(hex1, hex2, tol=1e-6):
        v1 = hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
        v2 = hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
        
        normals = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-15:
                    n = n / norm
                    normals.append(n)
        
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return True
        return False

    def point_in_hexagon(p, hex_verts, tol=1e-6):
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross < -tol:
                return False
        return True

    def compute_optimal_outer(verts, angle_deg):
        a = np.radians(angle_deg)
        normals = []
        for i in range(6):
            edge_angle = a + np.radians(60 * i)
            normal_angle = edge_angle + np.radians(90)
            normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
        normals = np.array(normals)
        
        projections = verts @ normals.T
        s_min = 0
        for i in range(6):
            span = projections[:, i].max() - projections[:, i].min()
            s_min = max(s_min, span / np.sqrt(3))
            
        mids = (projections.max(axis=0) + projections.min(axis=0)) / 2
        c = np.linalg.lstsq(normals, mids, rcond=None)[0]
        
        # Add margin to s to ensure containment with center c
        s_final = s_min * 1.05
        return s_final, c

    def is_valid(inner_arr, outer_arr, s_val):
        for i in range(12):
            for j in range(i + 1, 12):
                if not separated(inner_arr[i], inner_arr[j]):
                    return False
        outer_verts = hex_vertices(outer_arr[0], outer_arr[1], s_val, outer_arr[2])
        verts_check = np.vstack([hex_vertices(inner_arr[i, 0], inner_arr[i, 1], 1.0, inner_arr[i, 2]) for i in range(12)])
        for k in range(72):
            if not point_in_hexagon(verts_check[k], outer_verts):
                return False
        return True

    # Initial configuration: Scaled honeycomb grid
    sqrt3 = np.sqrt(3)
    positions = []
    for col in range(4):
        positions.append([col * sqrt3, 0, 0])
    for col in range(3):
        positions.append([(col + 0.5) * sqrt3, 1.5, 0])
    for col in range(4):
        positions.append([col * sqrt3, 3.0, 0])
    positions.append([sqrt3, 4.5, 0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    for p in positions:
        p[0] -= cx_mean
        p[1] -= cy_mean
        
    # Scale by 1.1 to ensure separation
    scale = 1.1
    inner_init = np.array(positions) * scale
    
    verts_init = np.vstack([hex_vertices(inner_init[i, 0], inner_init[i, 1], 1.0, inner_init[i, 2]) for i in range(12)])
    s_init, c_init = compute_optimal_outer(verts_init, 0)
    outer_init = np.array([c_init[0], c_init[1], 0.0])
    
    # Fallback is the scaled grid
    inner_fallback = inner_init.copy()
    outer_fallback = outer_init.copy()
    s_fallback = s_init
    
    best_inner = inner_fallback
    best_outer = outer_fallback
    best_s = s_fallback

    # Try optimization
    def objective(params):
        if time.time() > deadline - 30:
            return 1e6
        inner = params[:36].reshape(12, 3)
        outer_angle = params[36]
        
        # Check separation
        for i in range(12):
            for j in range(i + 1, 12):
                if not separated(inner[i], inner[j]):
                    return 1e6 + (j-i)
                    
        verts = np.vstack([hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(12)])
        s, c = compute_optimal_outer(verts, outer_angle)
        
        # Check containment
        outer_verts = hex_vertices(c[0], c[1], s, outer_angle)
        for k in range(72):
            if not point_in_hexagon(verts[k], outer_verts):
                return 1e6 + 1000 + k
                
        return s

    x0 = np.concatenate([inner_init.flatten(), [0.0]])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
        if result.fun < best_s and result.fun < 1e5:
            inner_opt = result.x[:36].reshape(12, 3)
            outer_angle = result.x[36]
            verts_opt = np.vstack([hex_vertices(inner_opt[i, 0], inner_opt[i, 1], 1.0, inner_opt[i, 2]) for i in range(12)])
            s_opt, c_opt = compute_optimal_outer(verts_opt, outer_angle)
            outer_opt = np.array([c_opt[0], c_opt[1], outer_angle])
            
            if is_valid(inner_opt, outer_opt, s_opt):
                best_inner = inner_opt
                best_outer = outer_opt
                best_s = s_opt
    except Exception:
        pass

    # Final check on best result
    if not is_valid(best_inner, best_outer, best_s):
        # If best is invalid, use fallback
        best_inner = inner_fallback
        best_outer = outer_fallback
        best_s = s_fallback
        if not is_valid(best_inner, best_outer, best_s):
             # Expand outer further
             best_s *= 1.2
             best_outer = outer_fallback.copy()
             best_outer[2] = 0.0

    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
