"""Rewritten program using smooth penalty-based optimization for hexagon packing."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    # Pre-compute constants
    HEX_NORMALS_BASE = np.array([
        [np.cos(np.deg2rad(30 + k*60)), np.sin(np.deg2rad(30 + k*60))]
        for k in range(6)
    ])
    HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + HEX_ANGLES
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon_fast(pts, cx, cy, side, angle_deg):
        """Check if array of points (n,2) are inside hexagon. Returns boolean array."""
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        apothem = side * np.sqrt(3) / 2
        normals = HEX_NORMALS_BASE
        dists = np.column_stack([local_x * normals[0,0] + local_y * normals[0,1],
                                  local_x * normals[1,0] + local_y * normals[1,1],
                                  local_x * normals[2,0] + local_y * normals[2,1],
                                  local_x * normals[3,0] + local_y * normals[3,1],
                                  local_x * normals[4,0] + local_y * normals[4,1],
                                  local_x * normals[5,0] + local_y * normals[5,1]])
        return np.all(dists <= apothem + 1e-9, axis=1)

    def sat_min_separation(v1, v2):
        """Compute minimum separation distance using SAT. Positive = separated."""
        edges1 = np.roll(v1, -1, axis=0) - v1
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals1 = np.column_stack([-edges1[:,1], edges1[:,0]])
        normals2 = np.column_stack([-edges2[:,1], edges2[:,0]])
        
        norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
        
        valid1 = norms1[:, 0] > 1e-9
        valid2 = norms2[:, 0] > 1e-9
        
        axes = []
        if np.any(valid1):
            axes.append(normals1[valid1] / norms1[valid1])
        if np.any(valid2):
            axes.append(normals2[valid2] / norms2[valid2])
        
        if not axes:
            return 0.0
            
        all_axes = np.vstack(axes)
        min_sep = float('inf')
        
        for axis in all_axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            sep = min(np.min(proj2) - np.max(proj1), np.min(proj1) - np.max(proj2))
            if sep > 0:
                return sep  # Already separated
            min_sep = min(min_sep, sep)
        
        return min_sep  # Negative if overlapping

    def required_outer_side(inner_positions, outer_angle_deg):
        """Compute minimal side length of outer hexagon centered at origin."""
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        angle_rad = np.radians(outer_angle_deg)
        apothem_factor = np.sqrt(3) / 2
        normal_angles = angle_rad + np.deg2rad(np.arange(30, 360, 60))
        
        max_side = 0
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dist = np.max(dists)
            if max_dist > 0:
                needed_side = max_dist / apothem_factor
                max_side = max(max_side, needed_side)
        
        return max_side

    def check_validity(inner, oc, os, oa):
        """Check if configuration is valid."""
        # Check all inside outer
        for i in range(len(inner)):
            cx, cy, angle = inner[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            if not np.all(point_in_hexagon_fast(verts, oc[0], oc[1], os, oa)):
                return False
        
        # Check pairwise disjoint
        verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in inner]
        for i in range(len(inner)):
            for j in range(i+1, len(inner)):
                if sat_min_separation(verts_list[i], verts_list[j]) < -1e-9:
                    return False
        return True

    def energy(params):
        """Smooth energy function for optimization."""
        inner_pos = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_pos.append((x, y, a))
        
        ox = params[3*N]
        oy = params[3*N+1]
        os = max(params[3*N+2], 0.1)
        oa = params[3*N+3]
        
        outer_center = (ox, oy)
        outer_side = os
        outer_angle = oa
        
        # Compute separation distances and containment violations
        verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in inner_pos]
        
        # Pairwise separation penalty (smooth)
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_min_separation(verts_list[i], verts_list[j])
                if sep < 0:
                    sep_penalty += sep * sep  # Quadratic penalty for overlap
        
        # Containment penalty (smooth)
        contain_penalty = 0.0
        for i in range(N):
            verts = verts_list[i]
            mask = ~point_in_hexagon_fast(verts, outer_center[0], outer_center[1], outer_side, outer_angle)
            if np.any(mask):
                # Compute how far outside
                dx = verts[mask, 0] - outer_center[0]
                dy = verts[mask, 1] - outer_center[1]
                angle_rad = np.radians(outer_angle)
                cos_a = np.cos(-angle_rad)
                sin_a = np.sin(-angle_rad)
                local_x = dx * cos_a - dy * sin_a
                local_y = dx * sin_a + dy * cos_a
                
                apothem = outer_side * np.sqrt(3) / 2
                normals = HEX_NORMALS_BASE
                for k in range(6):
                    dist = local_x * normals[k,0] + local_y * normals[k,1]
                    violation = np.maximum(0, dist - apothem)
                    contain_penalty += np.sum(violation ** 2)
        
        # Objective: minimize outer side + penalties
        return outer_side + sep_penalty * 10.0 + contain_penalty * 10.0

    def initial_config_hexagonal():
        """Create hexagonal cluster: 1 center + 6 first ring + 5 second ring."""
        positions = [(0.0, 0.0, 0.0)]
        
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.1 * np.cos(angle)
            y = 2.1 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        sqrt3 = np.sqrt(3)
        second_ring_candidates = [
            (4.2, 0.0),
            (3.15, 1.7325),
            (1.05, 3.465),
            (-1.05, 3.465),
            (-3.15, 1.7325),
            (-4.2, 0.0),
            (-3.15, -1.7325),
            (-1.05, -3.465),
            (1.05, -3.465),
            (3.15, -1.7325),
        ]
        
        from itertools import combinations
        best_combo = None
        best_side = float('inf')
        
        for combo in combinations(range(len(second_ring_candidates)), 5):
            test_positions = positions.copy()
            for idx in combo:
                x, y = second_ring_candidates[idx]
                test_positions.append((x, y, 0.0))
            side = required_outer_side(test_positions, 30.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        
        for idx in best_combo:
            x, y = second_ring_candidates[idx]
            positions.append((x, y, 0.0))
        
        return positions

    def initial_config_circular():
        """Create circular arrangement."""
        positions = []
        for k in range(N):
            angle = np.radians(360 * k / N)
            r = 3.0
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            positions.append((x, y, 0.0))
        return positions

    def initial_config_random():
        """Create random configuration."""
        positions = []
        for _ in range(N):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            a = np.random.uniform(0, 60)
            positions.append((x, y, a))
        return positions

    def make_params(inner, oc, os, oa):
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params, dtype=float)

    # Generate multiple initial configurations
    configs = [
        initial_config_hexagonal(),
        initial_config_circular(),
        initial_config_random(),
    ]
    
    # Find best initial config
    best_config = None
    best_side = float('inf')
    
    for inner_positions in configs:
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = required_outer_side(inner_positions, oa)
            if side < best_side:
                best_side = side
                best_config = (inner_positions, (0.0, 0.0), side, oa)
    
    inner_init, oc_init, os_init, oa_init = best_config
    
    # Ensure validity of initial config
    if not check_validity(inner_init, oc_init, os_init, oa_init):
        # Expand outer hexagon until valid
        while not check_validity(inner_init, oc_init, os_init, oa_init):
            os_init += 0.1
            if os_init > 15:
                break
    
    params_init = make_params(inner_init, oc_init, os_init, oa_init)
    best_params = params_init.copy()
    best_score = energy(params_init)
    
    # Simulated annealing
    def simulated_annealing(params, max_iters=5000, T0=1.0, T_end=0.001, alpha=0.9995):
        current = params.copy()
        current_energy = energy(current)
        best = current.copy()
        best_energy = current_energy
        
        T = T0
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Propose move
            new_params = current + np.random.normal(0, 0.1, size=len(current))
            new_energy = energy(new_params)
            
            # Accept with probability
            if new_energy < current_energy:
                current = new_params
                current_energy = new_energy
                if new_energy < best_energy:
                    best = current.copy()
                    best_energy = new_energy
            else:
                delta = new_energy - current_energy
                if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                    current = new_params
                    current_energy = new_energy
            
            T = max(T * alpha, T_end)
        
        return best, best_energy
    
    # Run simulated annealing with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            current_params = params_init.copy()
        else:
            # Add perturbation
            current_params = best_params + np.random.normal(0, 0.2, size=len(best_params))
        
        result_params, result_score = simulated_annealing(current_params, max_iters=3000)
        if result_score < best_score:
            best_score = result_score
            best_params = result_params
    
    # Final polish with Nelder-Mead
    if time.time() < deadline - 15 and best_score < 8.0:
        try:
            res = minimize(energy, best_params, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 0.001, 'fatol': 0.001, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Parse final result
    final_params = best_params
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append((x, y, a))
    oc = (final_params[3*N], final_params[3*N+1])
    os = final_params[3*N+2]
    oa = final_params[3*N+3]
    
    # Final validation with fallback
    if not check_validity(inner, oc, os, oa):
        inner = inner_init
        oc = oc_init
        os = os_init
        oa = oa_init
        while not check_validity(inner, oc, os, oa):
            os += 0.1
            if os > 15:
                break
    
    return inner, oc, os, oa
# EVOLVE-BLOCK-END
