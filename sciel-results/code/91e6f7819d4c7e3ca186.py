import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    sqrt3 = math.sqrt(3)
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * s1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * s2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        # Coarse scan
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
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

    def generate_honeycomb(seed_offset=0):
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        patterns = [
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(2,0),(-2,0),(1,-1),(-1,1),(0,2)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,-1),(-2,-1),(1,-1),(-1,-1),(0,2)],
        ]
        idx = (seed_offset) % len(patterns)
        lattice_coords = patterns[idx]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (li, lj) in enumerate(lattice_coords):
            positions[i] = li * a1 + lj * a2
            angles[i] = 0.0
        return positions, angles

    def generate_random_cluster():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        angles[0] = random.uniform(0, 60)
        for i in range(1, N):
            for attempt in range(50):
                ref_idx = random.randint(0, i-1)
                dx = random.uniform(-3.5, 3.5)
                dy = random.uniform(-3.5, 3.5)
                positions[i] = positions[ref_idx] + [dx, dy]
                angles[i] = random.uniform(0, 60)
                valid = True
                angles_rad = np.radians(angles[:i+1])
                h_new = (positions[i, 0], positions[i, 1], angles_rad[i])
                for j in range(i):
                    h_other = (positions[j, 0], positions[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h_new, h_other):
                        valid = False
                        break
                if valid:
                    break
            if not valid:
                positions[i] = [random.uniform(-5, 5), random.uniform(-5, 5)]
                angles[i] = random.uniform(0, 60)
        return positions, angles

    def smooth_objective(params, beta=20.0):
        """
        Smooth surrogate for the packing problem.
        Objective = SmoothOuterSide + Penalty * SmoothOverlap
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        
        # 1. Smooth Outer Side (Log-Sum-Exp of max projections)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # We need to minimize the outer side length.
        # The outer side is determined by the max projection over 6 directions.
        # Since we don't know the optimal orientation phi yet, we can either:
        # a) Assume phi is fixed (bad)
        # b) Include phi in params (good, but increases dimensionality)
        # c) Use a rotation-invariant smooth norm (approximate)
        
        # Let's include phi in the optimization variables for better accuracy.
        # params structure: [x1, y1, a1, ..., xN, yN, aN, phi]
        # Total dims: 3*N + 1
        
        phi = np.radians(params[-1])
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        
        # Soft max of side_vals
        beta_sm = 50.0
        m = np.max(side_vals)
        if m < 1e-10:
            smooth_side = m
        else:
            smooth_side = m + (1.0/beta_sm) * np.log(np.sum(np.exp(beta_sm * (side_vals - m))))
        
        # 2. Smooth Overlap Penalty
        # For each pair, compute the minimum distance. If distance < 0, penalty.
        # A simple smooth penalty: sum of max(0, overlap_depth)^2
        # Approximate overlap depth using support functions? 
        # Simpler: use a soft barrier based on center distance and rotation.
        # Since exact distance is hard to differentiate, we use a heuristic penalty:
        # Penalty = sum_{i<j} max(0, (R_min - dist_ij))^2
        # Where R_min is the minimum separation distance (approx 2*sqrt(3)/2? No, depends on angle).
        # For unit hexagons, min center distance for disjointness is roughly 1.7-2.0.
        # Let's use a conservative R_min = 1.8.
        
        penalty = 0.0
        R_min = 1.8
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < R_min:
                    # Smooth penalty
                    gap = R_min - dist
                    penalty += gap * gap
                    # Add a small derivative-friendly term if needed, but L-BFGS handles it via finite diff
        
        # Combine
        weight_penalty = 10.0
        return smooth_side + weight_penalty * penalty

    def polish_coordinate_descent(pos, ang, time_limit):
        """Refine using coordinate descent on the exact objective"""
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        
        t_start = time.time()
        step = 0.01
        min_step = 0.0001
        
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
                        
                        angles_rad_new = np.radians(new_ang)
                        valid = True
                        h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                        for j in range(N):
                            if j == idx: continue
                            h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                            if not hexagons_disjoint_fast(h_moved, h_other):
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
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                step *= 0.8
                continue
            break
        
        return best_side_cd, best_pos_cd, best_ang_cd

    # --- Main Optimization Loop ---
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Generate diverse starts
    starts = []
    for offset in range(3):
        pos, ang = generate_honeycomb(offset)
        starts.append((pos, ang))
    
    for seed in range(5):
        random.seed(seed * 999)
        pos, ang = generate_random_cluster()
        starts.append((pos, ang))
    
    # For each start, run L-BFGS-B on the smooth objective
    for i, (init_pos, init_ang) in enumerate(starts):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Initialize params: [x1, y1, a1, ..., xN, yN, aN, phi_init]
        # Guess phi_init from a quick scan
        init_side, init_phi, _, _ = compute_min_outer_side(init_pos, init_ang)
        
        init_params = np.zeros(3*N + 1)
        for j in range(N):
            init_params[3*j] = init_pos[j, 0]
            init_params[3*j+1] = init_pos[j, 1]
            init_params[3*j+2] = init_ang[j]
        init_params[-1] = init_phi
        
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for j in range(N):
                    new_pos[j, 0] = res.x[3*j]
                    new_pos[j, 1] = res.x[3*j+1]
                    new_ang[j] = res.x[3*j+2]
                new_phi = res.x[-1]
                
                # Check validity and compute exact side
                if is_valid_configuration(new_pos, new_ang):
                    exact_side, exact_phi, cx, cy = compute_min_outer_side(new_pos, new_ang)
                    if exact_side < best_side:
                        best_side = exact_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = exact_phi
                        best_cx = cx
                        best_cy = cy
        except Exception:
            pass

    # Polish the best result found
    if best_positions is not None:
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            cd_time = min(remaining_time * 0.8, 10.0)
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Fallback
    if best_positions is None or not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
