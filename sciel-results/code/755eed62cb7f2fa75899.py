"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def objective(params):
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        ox = params[3*n_inner]
        oy = params[3*n_inner+1]
        os = params[3*n_inner+2]
        oa = params[3*n_inner+3]
        
        outer_center = (ox, oy)
        outer_side = max(os, 0.1)
        outer_angle = oa
        
        inside_ok = all_inside_outer(inner_positions, outer_center, outer_side, outer_angle)
        disjoint_ok = pairwise_disjoint(inner_positions)
        
        if inside_ok and disjoint_ok:
            return outer_side
        
        penalty = 0.0
        if not inside_ok:
            for i in range(n_inner):
                cx, cy, angle = inner_positions[i]
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                        penalty += 1.0
        if not disjoint_ok:
            for i in range(n_inner):
                for j in range(i+1, n_inner):
                    ci = (inner_positions[i][0], inner_positions[i][1])
                    cj = (inner_positions[j][0], inner_positions[j][1])
                    ai = inner_positions[i][2]
                    aj = inner_positions[j][2]
                    v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                    v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                    if not polygons_disjoint_sat(v1, v2):
                        penalty += 1.0
                        
        return outer_side + penalty * 10.0

    def make_params(inner, oc, os, oa):
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return params

    def parse_params(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        oc = (params[3*N], params[3*N+1])
        os = params[3*N+2]
        oa = params[3*N+3]
        return inner, oc, os, oa

    # Initial valid configuration: Hexagonal lattice packing
    # 1 center, 6 in first ring, 5 in second ring (partial)
    # Or 4x3 grid as before but tighter
    
    x_coords = [-3.0, -1.0, 1.0, 3.0]
    y_coords = [-1.8, 0.0, 1.8]
    
    initial_positions = []
    for y in y_coords:
        for x in x_coords:
            initial_positions.append([x, y, 0.0])
            
    outer_side_init = 5.0
    outer_center_init = [0.0, 0.0]
    outer_angle_init = 30.0
    
    params = make_params(initial_positions, outer_center_init, outer_side_init, outer_angle_init)
    
    inner_init = [(p[0], p[1], p[2]) for p in initial_positions]
    if not is_valid(inner_init, outer_center_init, outer_side_init, outer_angle_init):
        outer_side_init = 6.0
        params = make_params(initial_positions, outer_center_init, outer_side_init, outer_angle_init)
        inner_init = [(p[0], p[1], p[2]) for p in initial_positions]
        if not is_valid(inner_init, outer_center_init, outer_side_init, outer_angle_init):
             while not all_inside_outer(inner_init, outer_center_init, outer_side_init, outer_angle_init):
                 outer_side_init += 0.1
                 if outer_side_init > 10: break
             params = make_params(initial_positions, outer_center_init, outer_side_init, outer_angle_init)

    best_params = params.copy()
    best_score = objective(params)
    
    # Optimization Loop with more aggressive strategy
    attempts = 0
    max_attempts = 8
    
    while time.time() < deadline - 10 and attempts < max_attempts:
        # Try different perturbation strategies
        if attempts < 3:
            # Small perturbations
            noise = np.random.uniform(-0.1, 0.1, size=len(params))
            current_params = best_params + noise
        else:
            # Larger perturbations for exploration
            noise = np.random.uniform(-0.5, 0.5, size=len(params))
            current_params = best_params + noise
            
        # Ensure outer side is reasonable
        if current_params[3*N+2] < 4.0:
            current_params[3*N+2] = 4.0
            
        try:
            res = minimize(objective, current_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
        attempts += 1
        
        # If we found a very good score, refine further
        if best_score < 4.5:
             try:
                 res2 = minimize(objective, best_params, method='Nelder-Mead', 
                                options={'maxiter': 300, 'xatol': 0.002, 'fatol': 0.002, 'adaptive': True})
                 if res2.fun < best_score:
                     best_score = res2.fun
                     best_params = res2.x
             except Exception:
                 pass
             break

    final_params = best_params
    inner, oc, os, oa = parse_params(final_params)
    
    # Final validation
    if not is_valid(inner, oc, os, oa):
        inner = [(p[0], p[1], p[2]) for p in initial_positions]
        oc = tuple(outer_center_init)
        os = outer_side_init
        oa = outer_angle_init
        if not is_valid(inner, oc, os, oa):
             while not all_inside_outer(inner, oc, os, oa):
                 os += 0.1
                 if os > 10: break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
