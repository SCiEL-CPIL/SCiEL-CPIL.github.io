"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Pre-compute constants
    HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))
    HEX_NORMALS_BASE = np.array([
        [np.cos(np.deg2rad(30 + k * 60)), np.sin(np.deg2rad(30 + k * 60))]
        for k in range(6)
    ])
    SQRT3 = np.sqrt(3)
    APOthem_factor = SQRT3 / 2  # apothem / side for unit hex

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + HEX_ANGLES
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_hex_normals(angle_deg):
        """Get the 6 outward normals of a hexagon with given rotation."""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        # Rotate base normals
        normals = HEX_NORMALS_BASE.copy()
        rotated = np.empty_like(normals)
        rotated[:, 0] = normals[:, 0] * cos_a - normals[:, 1] * sin_a
        rotated[:, 1] = normals[:, 0] * sin_a + normals[:, 1] * cos_a
        return rotated

    def min_distance_hex_hex(p1, a1, p2, a2):
        """
        Compute minimum distance between two unit hexagons.
        Returns 0 if they overlap or touch, positive if separated.
        Uses SAT: for each separating axis, compute gap.
        """
        v1 = hexagon_vertices(p1[0], p1[1], 1.0, a1)
        v2 = hexagon_vertices(p2[0], p2[1], 1.0, a2)
        
        # Get edge normals from both polygons
        axes = []
        for v in [v1, v2]:
            edges = np.roll(v, -1, axis=0) - v
            normals = np.column_stack([-edges[:, 1], edges[:, 0]])
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            valid = norms[:, 0] > 1e-9
            if np.any(valid):
                axes.append(normals[valid] / norms[valid])
        
        if not axes:
            return 0.0
        
        all_axes = np.vstack(axes)
        
        min_gap = 0.0
        for axis in all_axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            gap = min(max1 - min2, max2 - min1)
            if gap < 0:
                return 0.0  # Overlapping
            min_gap = max(min_gap, gap)
        
        return min_gap

    def compute_outer_side(inner_positions, outer_angle_deg):
        """
        Compute minimal side length of outer hexagon (centered at origin)
        that contains all inner hexagons, given outer rotation.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        angle_rad = np.radians(outer_angle_deg)
        normal_angles = angle_rad + np.deg2rad(np.arange(30, 360, 60))
        
        max_side = 0.0
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dist = np.max(dists)
            if max_dist > 0:
                needed_side = max_dist / APOthem_factor
                max_side = max(max_side, needed_side)
        
        return max_side

    def objective(params):
        """
        Smooth objective: outer_side + penalty for overlapping inner hexagons.
        """
        inner_pos = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            inner_pos.append((x, y, a))
        
        oa = params[3 * N]  # outer angle
        
        # Compute outer side (smooth, no penalty needed)
        outer_side = compute_outer_side(inner_pos, oa)
        
        # Penalty for overlapping inner hexagons
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_distance_hex_hex(inner_pos[i], inner_pos[i][2], 
                                            inner_pos[j], inner_pos[j][2])
                if dist < 0.05:
                    # Smooth penalty: quadratic for small distances
                    overlap_penalty += (0.05 - dist) ** 2 * 100.0
        
        return outer_side + overlap_penalty

    def make_initial_config_hex_cluster():
        """Hexagonal cluster: 1 center + 6 ring + 5 second ring."""
        positions = [(0.0, 0.0, 0.0)]
        
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        # Second ring: pick 5 from 10 candidates
        second_ring_candidates = [
            (4.0, 0.0),
            (3.0, SQRT3),
            (1.0, 2 * SQRT3),
            (-1.0, 2 * SQRT3),
            (-3.0, SQRT3),
            (-4.0, 0.0),
            (-3.0, -SQRT3),
            (-1.0, -2 * SQRT3),
            (1.0, -2 * SQRT3),
            (3.0, -SQRT3),
        ]
        
        from itertools import combinations
        best_combo = None
        best_side = float('inf')
        
        for combo in combinations(range(len(second_ring_candidates)), 5):
            test_positions = positions.copy()
            for idx in combo:
                x, y = second_ring_candidates[idx]
                test_positions.append((x, y, 0.0))
            side = compute_outer_side(test_positions, 30.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        
        for idx in best_combo:
            x, y = second_ring_candidates[idx]
            positions.append((x, y, 0.0))
        
        return positions

    def make_initial_config_staggered():
        """Staggered rows: 3 rows of 4."""
        positions = []
        for row in range(3):
            y = row * 2 * SQRT3 - 2 * SQRT3
            x_offset = (row % 2) * 2.0
            for col in range(4):
                x = col * 4.0 + x_offset - 6.0
                positions.append((x, y, 0.0))
        return positions

    def make_initial_config_symmetric():
        """Symmetric arrangement: 3x4 grid with slight rotation."""
        positions = []
        for i in range(3):
            for j in range(4):
                x = j * 3.8 - 5.7
                y = i * 3.4 - 3.4
                positions.append((x, y, 0.0))
        return positions

    # Generate initial configurations
    configs = []
    configs.append(make_initial_config_hex_cluster())
    configs.append(make_initial_config_staggered())
    configs.append(make_initial_config_symmetric())
    
    # Find best initial config
    best_init_params = None
    best_init_score = float('inf')
    
    for config in configs:
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            params = []
            for pos in config:
                params.extend([pos[0], pos[1], pos[2]])
            params.append(oa)
            params = np.array(params, dtype=float)
            
            score = objective(params)
            if score < best_init_score:
                best_init_score = score
                best_init_params = params.copy()
    
    best_params = best_init_params.copy()
    best_score = best_init_score
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        bounds = [(-8, 8)] * (3 * N) + [(0, 360)]
        
        def de_objective(params):
            if time.time() > deadline - 5:
                return float('inf')
            return objective(params)
        
        try:
            de_result = differential_evolution(
                de_objective, bounds,
                maxiter=50, popsize=15,
                tol=0.01, mutation=(0.3, 1.0),
                recombination=0.8, seed=42,
                polish=False, init='latinhypercube'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead local optimization
    if time.time() < deadline - 15:
        try:
            nm_result = minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_params = nm_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine polish
    if time.time() < deadline - 5 and best_score < 8.0:
        try:
            nm_result2 = minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 0.001, 'fatol': 0.001, 'adaptive': True}
            )
            if nm_result2.fun < best_score:
                best_score = nm_result2.fun
                best_params = nm_result2.x.copy()
        except Exception:
            pass
    
    # Parse final result
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        a = best_params[3 * i + 2]
        inner.append((x, y, a))
    
    oa = best_params[3 * N]
    outer_side = compute_outer_side(inner, oa)
    oc = (0.0, 0.0)
    
    # Validate: check all inside and disjoint
    def is_valid_final(inner, oc, os, oa):
        # Check inside
        outer_verts_normal_angles = np.radians(oa) + np.deg2rad(np.arange(30, 360, 60))
        apothem = os * APOthem_factor
        for cx, cy, angle in inner:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for na in outer_verts_normal_angles:
                nx = np.cos(na)
                ny = np.sin(na)
                dists = verts[:, 0] * nx + verts[:, 1] * ny
                if np.max(dists) > apothem + 1e-6:
                    return False
        # Check disjoint
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_distance_hex_hex(inner[i], inner[i][2], inner[j], inner[j][2])
                if dist < -1e-6:
                    return False
        return True
    
    if not is_valid_final(inner, oc, outer_side, oa):
        # Fallback: use initial config with expanded outer
        inner = configs[0]
        oa = 30.0
        outer_side = compute_outer_side(inner, oa)
        while not is_valid_final(inner, oc, outer_side, oa):
            outer_side += 0.05
            if outer_side > 10:
                break
    
    return inner, oc, outer_side, oa
# EVOLVE-BLOCK-END
