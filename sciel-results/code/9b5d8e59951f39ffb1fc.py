import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 10.0
    
    d = math.sqrt(3)
    
    # Generate base centers: center + ring 1 (6) + ring 2 (12)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
    
    ring2_centers = []
    ring2_indices = {}
    idx = 0
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
                ring2_indices[(q, r)] = idx
                idx += 1
    
    base_centers = centers[:7]
    n_base = len(base_centers)
    
    # Hexagon vertices relative to center (unit hexagon)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])
    
    def sat_distance(v1, v2):
        """Return minimum separation distance between two hexagons (negative if overlapping)."""
        min_gap = float('inf')
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
                min_gap = min(min_gap, gap)
        return min_gap
    
    def compute_S(params, alpha=0.0):
        """Compute the enclosing hexagon side length for a set of hexagons."""
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # Collect all vertices
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        k_angles = alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        dots = np.abs(np.sum(normals[:, np.newaxis, :] * all_pts[np.newaxis, :, :], axis=-1))
        max_per_normal = np.max(dots, axis=-1)
        max_proj = np.max(max_per_normal)
        return (2.0 / math.sqrt(3)) * max_proj
    
    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1e10
            
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # SAT overlap penalty
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                gap = sat_distance(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += gap * gap * 1000.0
        
        S = compute_S(params)
        return S + penalty
    
    # Generate all combinations of 4 from 12 ring 2 positions
    all_combos = list(itertools.combinations(range(12), 4))
    
    best_S = float('inf')
    best_params = None
    best_outer_alpha = 0.0
    
    # Phase 1: Coarse search over combinations
    combo_scores = []
    for combo in all_combos:
        final_centers = base_centers + [ring2_centers[i] for i in combo]
        params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers):
            params[3*i] = cx
            params[3*i+1] = cy
            params[3*i+2] = 0.0
        
        S_val = compute_S(params)
        
        # Check overlaps
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(N)]
        overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                if sat_distance(verts_list[i], verts_list[j]) < 0:
                    overlap = True
                    break
            if overlap:
                break
        
        if not overlap:
            combo_scores.append((S_val, combo))
    
    combo_scores.sort(key=lambda x: x[0])
    
    # Phase 2: Refine the top combos
    for S_coarse, combo in combo_scores[:15]:
        if time.time() > deadline - safety_margin:
            break
        
        final_centers = base_centers + [ring2_centers[i] for i in combo]
        init_params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers):
            init_params[3*i] = cx
            init_params[3*i+1] = cy
            init_params[3*i+2] = 0.0
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        
        except Exception:
            continue
    
    # If we have a valid best_params, optimize the outer hexagon rotation
    if best_params is not None:
        if time.time() < deadline - safety_margin:
            def outer_obj(alpha):
                return compute_S(best_params, alpha)
            
            try:
                res_outer = minimize_scalar(outer_obj, bounds=(0, math.pi/3), method='bounded',
                                           options={'xatol': 1e-8})
                if res_outer.success:
                    best_outer_alpha = res_outer.x
                    best_S = res_outer.fun
            except Exception:
                best_outer_alpha = 0.0
    else:
        # Fallback: use the best coarse combo without optimization
        if combo_scores:
            S_coarse, combo = combo_scores[0]
            final_centers = base_centers + [ring2_centers[i] for i in combo]
            best_params = np.zeros(N * 3)
            for i, (cx, cy) in enumerate(final_centers):
                best_params[3*i] = cx
                best_params[3*i+1] = cy
                best_params[3*i+2] = 0.0
            best_S = S_coarse
            best_outer_alpha = 0.0
        else:
            # Should not happen
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Construct the inner hexagons from best_params
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle_deg = math.degrees(best_params[3*i+2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_alpha)
