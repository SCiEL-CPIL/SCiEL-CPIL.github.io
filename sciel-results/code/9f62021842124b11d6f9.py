"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Pre-compute constants
    HEX_NORMALS_BASE = np.array([
        [np.cos(np.deg2rad(30 + k*60)), np.sin(np.deg2rad(30 + k*60))]
        for k in range(6)
    ])
    HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))
    SQRT3 = np.sqrt(3)
    APOTHEM_FACTOR = SQRT3 / 2.0

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

        apothem = side * APOTHEM_FACTOR
        # normals shape (6,2), local_x/y shape (n,)
        dists = local_x[:, np.newaxis] * HEX_NORMALS_BASE[:, 0] + local_y[:, np.newaxis] * HEX_NORMALS_BASE[:, 1]
        # dists shape (n, 6)
        return np.all(dists <= apothem + 1e-9, axis=1)

    def polygons_disjoint_fast(v1, v2):
        """SAT check using precomputed edge normals. v1, v2 are (6,2) arrays."""
        # Get edge normals from both polygons
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
            return False
            
        all_axes = np.vstack(axes)
        
        for axis in all_axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                return True
        return False

    def all_inside_outer_fast(inner_positions, outer_center, outer_side, outer_angle):
        """Check all inner hex vertices are inside outer hexagon."""
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            if not np.all(point_in_hexagon_fast(verts, outer_center[0], outer_center[1], outer_side, outer_angle)):
                return False
        return True

    def pairwise_disjoint_fast(inner_positions):
        """Check all pairs of inner hexagons are disjoint."""
        verts_list = []
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts_list.append(hexagon_vertices(cx, cy, 1.0, angle))
        
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                if not polygons_disjoint_fast(verts_list[i], verts_list[j]):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside_outer_fast(inner, oc, os, oa) and pairwise_disjoint_fast(inner)

    def compute_minimal_outer(inner_positions, outer_angle_deg):
        """Compute minimal side length of outer hexagon centered at origin."""
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        angle_rad = np.radians(outer_angle_deg)
        normal_angles = angle_rad + np.deg2rad(np.arange(30, 360, 60))
        
        max_side = 0
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dist = np.max(dists)
            if max_dist > 0:
                needed_side = max_dist / APOTHEM_FACTOR
                max_side = max(max_side, needed_side)
        
        return max_side

    def make_hex_cluster():
        """Create hexagonal cluster: 1 center + 6 first ring + 5 second ring."""
        positions = [(0.0, 0.0, 0.0)]
        
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        sqrt3 = SQRT3
        second_ring_candidates = [
            (4.0, 0.0),
            (3.0, sqrt3),
            (1.0, 2*sqrt3),
            (-1.0, 2*sqrt3),
            (-3.0, sqrt3),
            (-4.0, 0.0),
            (-3.0, -sqrt3),
            (-1.0, -2*sqrt3),
            (1.0, -2*sqrt3),
            (3.0, -sqrt3),
        ]
        
        best_combo = None
        best_side = float('inf')
        
        for combo in combinations(range(len(second_ring_candidates)), 5):
            test_positions = positions.copy()
            for idx in combo:
                x, y = second_ring_candidates[idx]
                test_positions.append((x, y, 0.0))
            side = compute_minimal_outer(test_positions, 30.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        
        for idx in best_combo:
            x, y = second_ring_candidates[idx]
            positions.append((x, y, 0.0))
        
        return positions, best_side

    def make_staggered_rows():
        """Create staggered row configuration."""
        positions = []
        sqrt3 = SQRT3
        # 4 rows of 3 hexagons, staggered
        for row in range(4):
            y_offset = row * 2 * sqrt3
            x_offset = (row % 2) * 2.0
            for col in range(3):
                x = col * 4.0 + x_offset - 4.0
                y = y_offset - 3 * sqrt3
                positions.append((x, y, 0.0))
        return positions

    def make_grid_config():
        """Create grid-like configuration."""
        positions = []
        sqrt3 = SQRT3
        # 3x4 grid
        for i in range(3):
            for j in range(4):
                x = j * 4.0 - 6.0
                y = i * 2 * sqrt3 - 2 * sqrt3
                positions.append((x, y, 0.0))
        return positions

    def make_compact_hex():
        """Create a more compact hexagonal arrangement."""
        positions = [(0.0, 0.0, 0.0)]
        # First ring: 6 hexagons at distance 2
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        # Second ring: pick 5 from 12 positions
        sqrt3 = SQRT3
        candidates = [
            (4.0, 0.0), (3.0, sqrt3), (1.0, 2*sqrt3),
            (-1.0, 2*sqrt3), (-3.0, sqrt3), (-4.0, 0.0),
            (-3.0, -sqrt3), (-1.0, -2*sqrt3), (1.0, -2*sqrt3),
            (3.0, -sqrt3), (2.0, 0.0), (-2.0, 0.0)
        ]
        best_combo = None
        best_side = float('inf')
        for combo in combinations(range(len(candidates)), 5):
            test_positions = positions.copy()
            for idx in combo:
                x, y = candidates[idx]
                test_positions.append((x, y, 0.0))
            side = compute_minimal_outer(test_positions, 30.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        for idx in best_combo:
            x, y = candidates[idx]
            positions.append((x, y, 0.0))
        return positions

    # Generate multiple initial configurations
    configs = []
    
    # Hex cluster
    hex_pos, hex_side = make_hex_cluster()
    configs.append(hex_pos)
    
    # Staggered rows
    configs.append(make_staggered_rows())
    
    # Grid
    configs.append(make_grid_config())
    
    # Compact hex
    configs.append(make_compact_hex())
    
    # Find best initial config
    best_config = None
    best_side = float('inf')
    
    for inner_positions in configs:
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = compute_minimal_outer(inner_positions, oa)
            if side < best_side:
                best_side = side
                best_config = (inner_positions, (0.0, 0.0), side, oa)
    
    inner_init, oc_init, os_init, oa_init = best_config
    
    # Verify and expand if needed
    if not is_valid(inner_init, oc_init, os_init, oa_init):
        while not all_inside_outer_fast(inner_init, oc_init, os_init, oa_init):
            os_init += 0.05
            if os_init > 10:
                break

    def objective(params):
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
        
        inside_ok = all_inside_outer_fast(inner_pos, outer_center, outer_side, outer_angle)
        disjoint_ok = pairwise_disjoint_fast(inner_pos)
        
        if inside_ok and disjoint_ok:
            return outer_side
        
        penalty = 0.0
        if not inside_ok:
            for i in range(N):
                cx, cy, angle = inner_pos[i]
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                mask = ~point_in_hexagon_fast(verts, outer_center[0], outer_center[1], outer_side, outer_angle)
                penalty += np.sum(mask)
        if not disjoint_ok:
            verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in inner_pos]
            for i in range(N):
                for j in range(i+1, N):
                    if not polygons_disjoint_fast(verts_list[i], verts_list[j]):
                        penalty += 1.0
                        
        return outer_side + penalty * 10.0

    def make_params(inner, oc, os, oa):
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params, dtype=float)

    params_init = make_params(inner_init, oc_init, os_init, oa_init)
    best_params = params_init.copy()
    best_score = objective(params_init)
    
    # Local optimization with multiple restarts
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        if restart == 0:
            current_params = params_init.copy()
        else:
            # Add small random perturbation
            current_params = best_params + np.random.normal(0, 0.05, size=len(best_params))
        
        try:
            res = minimize(objective, current_params, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.003, 'fatol': 0.003, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10 and best_score < 8.0:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 0.0005, 'fatol': 0.0005, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
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
    if not is_valid(inner, oc, os, oa):
        inner = inner_init
        oc = oc_init
        os = os_init
        oa = oa_init
        while not all_inside_outer_fast(inner, oc, os, oa):
            os += 0.05
            if os > 10:
                break
    
    return inner, oc, os, oa
# EVOLVE-BLOCK-END
