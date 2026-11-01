"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        v1, v2: Nx2 arrays of vertices.
        Returns True if disjoint, False if overlapping.
        """
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
        os = max(params[3*n_inner+2], 0.1)
        oa = params[3*n_inner+3]
        
        outer_center = (ox, oy)
        outer_side = os
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
        return np.array(params, dtype=float)

    def parse_params(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        oc = (params[3*N], params[3*N+1])
        os = params[3*N+2]
        oa = params[3*N+3]
        return inner, oc, os, oa

    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: 4x3 grid (current)
    x_coords = [-3.0, -1.0, 1.0, 3.0]
    y_coords = [-1.8, 0.0, 1.8]
    config1 = []
    for y in y_coords:
        for x in x_coords:
            config1.append([x, y, 0.0])
    initial_configs.append((config1, [0.0, 0.0], 5.0, 30.0))
    
    # Config 2: Hexagonal lattice - 1 center + 6 ring + 5 more
    # Center at origin
    config2 = [[0.0, 0.0, 0.0]]
    # Ring of 6 at distance ~2.0
    for k in range(6):
        angle = np.radians(60 * k)
        config2.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
    # Add 5 more in second ring
    for k in range(5):
        angle = np.radians(60 * k + 30)
        config2.append([3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0])
    initial_configs.append((config2, [0.0, 0.0], 5.5, 30.0))
    
    # Config 3: 2 rows of 6, staggered
    config3 = []
    for i in range(6):
        config3.append([i * 2.0 - 5.0, -0.9, 0.0])
        config3.append([i * 2.0 - 4.0, 0.9, 0.0])
    initial_configs.append((config3, [0.0, 0.0], 6.0, 30.0))
    
    # Find best initial config
    best_params = None
    best_score = float('inf')
    
    for config, oc_init, os_init, oa_init in initial_configs:
        params = make_params(config, oc_init, os_init, oa_init)
        inner_init = [(p[0], p[1], p[2]) for p in config]
        
        # Ensure validity by expanding if needed
        os_try = os_init
        while not all_inside_outer(inner_init, oc_init, os_try, oa_init):
            os_try += 0.1
            if os_try > 10:
                break
        params = make_params(config, oc_init, os_try, oa_init)
        
        score = objective(params)
        if score < best_score:
            best_score = score
            best_params = params
    
    # Optimization with multiple restarts
    np.random.seed(42)
    attempts = 0
    max_attempts = 8
    
    while time.time() < deadline - 15 and attempts < max_attempts:
        # Create perturbed start
        if attempts == 0:
            current_params = best_params.copy()
        else:
            noise = np.random.uniform(-0.3, 0.3, size=len(best_params))
            current_params = best_params + noise
            # Keep outer side reasonable
            if current_params[3*N+2] < 4.0:
                current_params[3*N+2] = 4.0
        
        try:
            # Use Nelder-Mead for robustness
            res = minimize(objective, current_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        attempts += 1
        
        # Early termination if good solution found
        if best_score < 4.2:
            break

    # Final refinement if we have time
    if time.time() < deadline - 5 and best_score < 4.5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 150, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    final_params = best_params
    inner, oc, os, oa = parse_params(final_params)
    
    # Final validation and fallback
    if not is_valid(inner, oc, os, oa):
        # Use the best initial config as fallback
        config, oc_init, os_init, oa_init = initial_configs[0]
        inner = [(p[0], p[1], p[2]) for p in config]
        oc = tuple(oc_init)
        os = os_init
        oa = oa_init
        
        # Expand if needed
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.1
            if os > 10:
                break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
