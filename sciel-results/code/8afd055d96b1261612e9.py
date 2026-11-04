import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 143.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# Minkowski sum of two unit hexagons is a hexagon of side 2.
# The Minkowski sum's edge normals are at angles 30° + k*60° (midpoints of unit hex normals).
# Non-overlap: for all 6 Minkowski normals m_i, |dot(delta, m_i)| >= 2
# Violation: max(0, 2 - max_i |dot(delta, m_i)|)
_MINK_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ang = np.radians(30 + _i * 60)
    _MINK_NORMALS[_i] = [math.cos(_ang), math.sin(_ang)]

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]
    
    # Precompute the 6 Minkowski sum edge normal vectors
    mn_x = _MINK_NORMALS[:, 0]  # (6,)
    mn_y = _MINK_NORMALS[:, 1]  # (6,)
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)
    
    def minkowski_violation(positions, angles_deg):
        """Compute Minkowski sum collision violation for all pairs.
        Returns array of shape (N*(N-1)/2,) with violation amounts.
        Violation > 0 means overlap.
        The Minkowski sum of two unit hexagons (regardless of relative rotation)
        is a regular hexagon of side 2 with normals at 30°+k*60°.
        Non-overlap iff max_i |dot(delta, m_i)| >= 2 for all 6 Minkowski normals."""
        n_pairs = 0
        violations = np.zeros(N*(N-1)//2)
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dots = dx * mn_x + dy * mn_y  # (6,)
                abs_dots = np.abs(dots)
                max_dot = np.max(abs_dots)
                viol = max(0.0, 2.0 - max_dot)
                violations[n_pairs] = viol
                n_pairs += 1
        return violations
    
    def minkowski_max_violation(positions, angles_deg):
        """Return the maximum violation (0 if no overlap)."""
        v = minkowski_violation(positions, angles_deg)
        return np.max(v)
    
    def hexagons_disjoint_fast(h1, h2):
        """Check if two unit hexagons are disjoint using Minkowski sum criterion.
        h1, h2: (x, y, angle_rad)
        Two convex polygons A and B are disjoint iff there exists a direction
        where their projections don't overlap. For two regular hexagons, the
        Minkowski sum A+(-B) is a regular hexagon of side 2 (independent of
        relative rotation due to 6-fold symmetry). So A and B are disjoint
        iff the center-to-center vector delta is outside this Minkowski hexagon,
        i.e., max_i |dot(delta, m_i)| >= 2 where m_i are the 6 Minkowski normals."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        # Quick reject: if center distance > circumradius of Minkowski hex (2)
        # The Minkowski hex has circumradius 2, so if |delta| > 2, definitely disjoint
        # Actually the inradius is sqrt(3), circumradius is 2
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.0:
            return True
        # Project onto all 6 Minkowski normals
        dots = dx * mn_x + dy * mn_y
        abs_dots = np.abs(dots)
        max_dot = np.max(abs_dots)
        return max_dot >= 2.0
    
    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]
    
    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True
    
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    def make_honeycomb():
        # Honeycomb packing: each hexagon touches 6 neighbors
        # Center-to-center distance for touching unit hexagons = 2*sqrt(3) ≈ 3.464
        # Lattice vectors: a1 = (2*sqrt(3), 0), a2 = (sqrt(3), 3)
        # Wait, let me reconsider. For unit side hexagons:
        # Width (flat-to-flat) = 2*sqrt(3), height (point-to-point) = 2
        # In a honeycomb, centers are spaced by 2*sqrt(3) in the horizontal direction
        # and by 3 in the vertical direction with sqrt(3) horizontal offset
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang
    
    def make_ring_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 6] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang
    
    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 1.5 * 2]
        return pos, ang
    
    def make_compact_hex():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang
    
    init_configs = [make_honeycomb(), make_ring_config(), make_two_rows(), make_compact_hex()]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
    
    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Phase 1: SA for global exploration (45% of time)
    sa_time_budget = DEADLINE * 0.45
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    T = 0.15
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.15
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 3.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        # Fast validity check: only check moved hexagon against others
        valid = True
        new_ang_rad = math.radians(new_angles[idx])
        for j in range(N):
            if j == idx:
                continue
            other_ang_rad = math.radians(current_angles[j])
            if not hexagons_disjoint_fast(
                (new_positions[idx, 0], new_positions[idx, 1], new_ang_rad),
                (current_positions[j, 0], current_positions[j, 1], other_ang_rad)
            ):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
            continue
        
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        
        if total_count % 200 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.05:
                step_size *= 0.85
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.35:
                step_size *= 1.12
                if step_size > 0.5: step_size = 0.5
            accept_count = 0
            total_count = 0
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003
    
    # Phase 2: Smooth L-BFGS-B optimization from multiple starts (35% of time)
    smooth_time_budget = DEADLINE * 0.35
    
    def smooth_objective(params):
        """Smooth objective: outer side + penalty for Minkowski collision violations."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        # Compute outer side
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Use current best phi for smoothness, or a fixed reasonable one
        phi = np.radians(best_phi)
        na = phi + np.pi/6 + np.arange(6) * np.pi/3
        nxx = np.cos(na)
        nyy = np.sin(na)
        pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
        abs_pr = np.abs(pr)
        
        # Log-sum-exp smooth max
        beta = 30.0
        max_val = np.max(abs_pr, axis=0)
        smooth_max = max_val + np.log(np.mean(np.exp((abs_pr - max_val[:, None]) * beta), axis=0)) / beta
        outer_side = (2.0 / sqrt3) * np.max(smooth_max)
        
        # Minkowski collision penalty
        violations = minkowski_violation(pos, ang)
        # Smooth penalty: sum of exp(violation * gamma) - 1, or just sum of violations
        # Use a smooth quadratic-like penalty
        gamma = 5.0
        penalty = np.sum(np.exp(violations * gamma) - 1.0)
        
        return outer_side + 10.0 * penalty
    
    def smooth_objective_grad(params):
        """Gradient of smooth objective (numerical)."""
        eps = 1e-7
        grad = np.zeros(len(params))
        f0 = smooth_objective(params)
        for i in range(len(params)):
            params[i] += eps
            f1 = smooth_objective(params)
            params[i] -= eps
            grad[i] = (f1 - f0) / (2 * eps)
        return grad
    
    # Generate multiple starting points for L-BFGS-B
    start_configs = [best_positions.copy(), best_angles.copy()]
    
    # Add perturbed versions
    for _ in range(3):
        p = best_positions.copy()
        a = best_angles.copy()
        for i in range(N):
            p[i, 0] += random.gauss(0, 0.3)
            p[i, 1] += random.gauss(0, 0.3)
            a[i] += random.gauss(0, 5.0)
        start_configs.append(p)
        start_configs.append(a)
    
    # Also try the initial configs
    for pos0, ang0 in init_configs[:2]:
        start_configs.append(pos0.copy())
        start_configs.append(ang0.copy())
    
    remaining_for_smooth = smooth_time_budget - (time.time() - t0)
    if remaining_for_smooth > 3.0:
        num_starts = min(len(start_configs)//2, max(2, int(remaining_for_smooth / 8.0)))
        
        for s_idx in range(num_starts):
            if time.time() - t0 > DEADLINE - 5.0:
                break
            
            pos_start = start_configs[2*s_idx]
            ang_start = start_configs[2*s_idx+1]
            
            # Check validity
            if not is_valid_configuration(pos_start, ang_start):
                continue
            
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = pos_start[i, 0]
                init_params[3*i+1] = pos_start[i, 1]
                init_params[3*i+2] = ang_start[i]
            
            try:
                res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
                
                if res.fun < best_side + 1.0:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass
    
    # Phase 3: Coordinate descent polish (20% of time)
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 2.0:
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.02
            min_step = 0.0002
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step
                            
                            # Fast validity: only check moved hexagon
                            new_ang_rad = math.radians(new_ang[idx])
                            valid = True
                            for j in range(N):
                                if j == idx: continue
                                other_ang_rad = math.radians(new_ang[j])
                                if not hexagons_disjoint_fast(
                                    (new_pos[idx, 0], new_pos[idx, 1], new_ang_rad),
                                    (new_pos[j, 0], new_pos[j, 1], other_ang_rad)
                                ):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                            
                            if new_side < best_side_cd - 1e-10:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                            
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.6
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd
        
        cd_time = min(remaining_time * 0.6, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # Second round of CD with even finer steps
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            cd_time2 = min(remaining_time * 0.4, 8.0)
            cd_side2, cd_pos2, cd_ang2 = polish_coordinate_descent(best_positions, best_angles, cd_time2)
            if cd_side2 < best_side:
                best_side = cd_side2
                best_positions = cd_pos2
                best_angles = cd_ang2
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # Final phi refinement
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 1.0:
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            def eval_phi_fine(phi_deg):
                phi = np.radians(phi_deg)
                na = phi + np.pi/6 + np.arange(6) * np.pi/3
                nxx = np.cos(na)
                nyy = np.sin(na)
                pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
                return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
            
            try:
                lo = max(0.0, best_phi - 3.0)
                hi = min(60.0, best_phi + 3.0)
                res = minimize_scalar(eval_phi_fine, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-12})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
                    best_cx = center[0]
                    best_cy = center[1]
            except Exception:
                pass
    
    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Safety: ensure we're well within deadline
    if time.time() - t0 > DEADLINE - 2.0:
        pass  # Just return what we have
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
