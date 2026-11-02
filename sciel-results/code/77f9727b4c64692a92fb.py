"""
Strategy: Simulated Annealing with direct outer-hexagon minimization.
Instead of optimizing the outer hexagon parameters alongside the inner ones,
we treat the outer hexagon as the minimal container for the current inner
configuration. This reduces the dimensionality and allows the SA to focus
on packing the inner hexagons tightly. The outer side length is computed
by scanning the rotation angle of the outer hexagon to find the minimum
size that contains all inner hexagons.
"""
import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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

    def compute_minimal_outer_side(inner_positions, outer_center, outer_angle):
        """
        Given fixed inner positions and outer center/angle, find the minimal
        outer side length such that all inner hexagons fit inside.
        """
        # For a regular hexagon with center (cx, cy), side s, angle a,
        # a point (px, py) is inside iff for all 6 normals n_k:
        #   (px-cx, py-cy) . n_k <= s * sqrt(3)/2
        # So s >= max over all points and all normals of ((px-cx, py-cy).n_k) / (sqrt(3)/2)
        
        apothem_factor = np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        angle_rad = np.radians(outer_angle)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        max_required = 0.0
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                dx = vx - outer_center[0]
                dy = vy - outer_center[1]
                local_x = dx * cos_a - dy * sin_a
                local_y = dx * sin_a + dy * cos_a
                dists = local_x * normals[:, 0] + local_y * normals[:, 1]
                max_required = max(max_required, np.max(dists))
        
        return max_required / apothem_factor

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
        oa = params[3*n_inner+2]
        
        # Check disjointness
        if not pairwise_disjoint(inner_positions):
            # Penalty for overlapping
            penalty = 0.0
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
            return 100.0 + penalty * 10.0
        
        # Compute minimal outer side
        os = compute_minimal_outer_side(inner_positions, (ox, oy), oa)
        return os

    def make_params(inner, oc, oa):
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], oa])
        return np.array(params, dtype=float)

    def parse_params(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        oc = (params[3*N], params[3*N+1])
        oa = params[3*N+2]
        return inner, oc, oa

    # Generate initial configuration
    # Start with a hexagonal packing
    config = []
    # Center hexagon
    config.append([0.0, 0.0, 0.0])
    # Ring of 6
    for k in range(6):
        angle = np.radians(60 * k)
        config.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
    # Ring of 5 more (partial second ring)
    for k in range(5):
        angle = np.radians(60 * k + 30)
        config.append([3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0])
    
    oc_init = [0.0, 0.0]
    oa_init = 30.0
    
    params = make_params(config, oc_init, oa_init)
    
    # Simulated Annealing
    np.random.seed(42)
    best_params = params.copy()
    best_score = objective(best_params)
    
    current_params = params.copy()
    current_score = best_score
    
    T = 2.0
    T_min = 0.001
    cooling_rate = 0.995
    step_size = 1.0
    
    max_iters = 50000
    iter_count = 0
    
    while T > T_min and time.time() < deadline - 10:
        # Generate new candidate
        new_params = current_params + np.random.uniform(-step_size, step_size, size=len(current_params))
        
        # Keep angles in reasonable range
        for i in range(N):
            new_params[3*i+2] = np.mod(new_params[3*i+2], 360)
        new_params[3*N+2] = np.mod(new_params[3*N+2], 360)
        
        new_score = objective(new_params)
        
        # Accept or reject
        if new_score < current_score or np.random.random() < math.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            
            if new_score < best_score:
                best_params = new_params.copy()
                best_score = new_score
        
        # Cool down
        T *= cooling_rate
        iter_count += 1
        
        # Reduce step size gradually
        if iter_count % 1000 == 0:
            step_size *= 0.95
    
    # Final refinement
    if time.time() < deadline - 5:
        step_size = 0.1
        T = 0.1
        T_min = 0.0001
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            new_params = current_params + np.random.uniform(-step_size, step_size, size=len(current_params))
            for i in range(N):
                new_params[3*i+2] = np.mod(new_params[3*i+2], 360)
            new_params[3*N+2] = np.mod(new_params[3*N+2], 360)
            
            new_score = objective(new_params)
            
            if new_score < current_score or np.random.random() < math.exp(-(new_score - current_score) / T):
                current_params = new_params
                current_score = new_score
                
                if new_score < best_score:
                    best_params = new_params.copy()
                    best_score = new_score
            
            T *= 0.999
    
    final_params = best_params
    inner, oc, oa = parse_params(final_params)
    
    # Compute final outer side
    os = compute_minimal_outer_side(inner, oc, oa)
    
    # Final validation
    if not is_valid(inner, oc, os, oa):
        # Fallback: use a known valid configuration
        config_fallback = []
        x_coords = [-3.0, -1.0, 1.0, 3.0]
        y_coords = [-1.8, 0.0, 1.8]
        for y in y_coords:
            for x in x_coords:
                config_fallback.append([x, y, 0.0])
        inner = [(p[0], p[1], p[2]) for p in config_fallback]
        oc = (0.0, 0.0)
        oa = 30.0
        os = compute_minimal_outer_side(inner, oc, oa)
        # Expand if needed
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.1
            if os > 10:
                break

    return inner, oc, os, oa
