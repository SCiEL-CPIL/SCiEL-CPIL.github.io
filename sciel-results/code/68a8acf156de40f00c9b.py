import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 143.0
SQRT3 = math.sqrt(3)

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
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = SQRT3
    
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
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
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

        phi_vals = np.radians(np.arange(0, 60, 0.5))
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
        best_phi = float(phi_vals[best_idx])

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
    
    # Add some random valid-ish starts
    for _ in range(5):
        pos = np.random.randn(N, 2) * 2.0
        ang = np.random.randn(N) * 10.0
        init_configs.append((pos, ang))

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate initial configs
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

    # Define smooth objective for L-BFGS-B
    # We optimize over positions and angles (36 params)
    # Objective: Smooth Outer Side + Penalty for Overlaps
    
    def smooth_objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        # 1. Check overlaps (soft penalty)
        # Calculate min distance for each pair? 
        # Simpler: use a soft penalty based on the sum of squared overlaps if we had a distance metric.
        # Here we use a heuristic: if any pair is close, penalize.
        # To make it differentiable, we can use the sum of squared "invasions" if we defined them,
        # but exact hexagon overlap is hard. 
        # Instead, we rely on the fact that L-BFGS-B will push them apart if the side length increases.
        # However, without a gradient for overlap, it might get stuck in overlaps.
        # Let's add a simple penalty: sum of exp(-dist) for pairs closer than 2*sqrt3 (approx diameter).
        # This is smooth and pushes them apart.
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Hexagon "radius" is 1, so distance < 2 is definitely overlapping centers? No, side 1 -> width 2*sqrt(3)/2 * 2?
                # Width of hexagon is 2*sqrt(3)/2 * 2? No.
                # Flat-to-flat distance is sqrt(3). Point-to-point is 2.
                # If centers are closer than ~1.5, they are likely overlapping.
                if dist < 2.0:
                    penalty += (2.0 - dist)**2
        
        if penalty > 100:
            return penalty + 1000.0 # Hard fail
            
        # 2. Compute smooth outer side
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Use a fixed phi for the gradient step to make it differentiable?
        # Or optimize phi too? 
        # If we fix phi, we might miss the optimal rotation.
        # Let's include phi in the optimization parameters? 
        # Total params: 3*N + 1 = 37.
        # But the prompt says return angle. 
        # Let's stick to optimizing positions and angles of inner hexes, and find optimal outer phi in a post-step.
        # For the smooth objective, we can use a soft-max over the 6 directions for a *fixed* phi?
        # No, the outer hexagon can rotate. 
        # If we fix phi in the objective, the gradient will push points to minimize the projection for *that* phi.
        # This might lead to a local minimum for that phi.
        # Better: Use the current best_phi as the fixed direction for the smooth objective?
        # Or just use a fixed phi=0 for the smooth phase, and let the final polish handle the rotation?
        # The rotation of the outer hexagon is a global property.
        # Let's try optimizing positions/angles with phi fixed to 0 (or current best) in the smooth phase.
        
        phi = np.radians(best_phi) # Use current best phi as a guide
        na = phi + np.pi/6 + np.arange(6) * np.pi/3
        nxx = np.cos(na)
        nyy = np.sin(na)
        pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
        abs_pr = np.abs(pr)
        
        # Log-sum-exp approximation of max
        beta = 50.0
        max_val = np.max(abs_pr, axis=0)
        # Soft max of the 6 projections
        smooth_max = max_val + np.log(np.mean(np.exp((abs_pr - max_val[:, None]) * beta), axis=0)) / beta
        
        side = (2.0 / sqrt3) * np.max(smooth_max)
        
        return side + 10.0 * penalty

    # Multi-start L-BFGS-B
    t_start_opt = time.time()
    time_budget_opt = 60.0 # 60 seconds for optimization
    
    # We will run L-BFGS-B from the best initial config and a few random ones
    start_configs = [best_positions.copy(), best_angles.copy()]
    
    # Add a few random perturbations of the best config
    for _ in range(3):
        p = best_positions.copy()
        a = best_angles.copy()
        p += np.random.randn(N, 2) * 0.5
        a += np.random.randn(N) * 5.0
        start_configs.append((p, a))
        
    # Note: start_configs is a list of (pos, ang) tuples, but I mixed it up.
    # Let's restructure.
    
    candidates = [(best_positions, best_angles)]
    for _ in range(3):
        p = best_positions.copy()
        a = best_angles.copy()
        p += np.random.randn(N, 2) * 0.5
        a += np.random.randn(N) * 5.0
        candidates.append((p, a))
        
    for p0, a0 in candidates:
        if time.time() - t_start_opt > time_budget_opt:
            break
            
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = p0[i, 0]
            init_params[3*i+1] = p0[i, 1]
            init_params[3*i+2] = a0[i]
            
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6})
            
            if res.fun < 100: # Check if it didn't explode
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

    # Phase 2: High precision polish using Nelder-Mead on exact objective
    if time.time() - t0 < DEADLINE - 10:
        def exact_objective(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            
            if not is_valid_configuration(pos, ang):
                return 1000.0 + np.random.rand()
            
            s, _, _, _ = compute_min_outer_side(pos, ang)
            return s

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
            
        try:
            # Nelder-Mead is good for final polishing of nonsmooth functions
            res = minimize(exact_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_side + 0.1:
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

    # Final safety check
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
    
    try:
        angles_rad_final = np.radians(best_angles)
        all_verts_final = get_all_vertices(best_positions, angles_rad_final)
        verts_flat_final = all_verts_final.reshape(-1, 2)
        center_final = np.mean(verts_flat_final, axis=0)
        offsets_final = verts_flat_final - center_final
        
        def exact_side_for_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets_final[:, 0:1] * nxx[None, :] + offsets_final[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
        
        phi_grid = np.linspace(max(0, best_phi - 0.5), min(60, best_phi + 0.5), 101)
        sides_grid = np.array([exact_side_for_phi(p) for p in phi_grid])
        best_idx = np.argmin(sides_grid)
        if sides_grid[best_idx] < best_side - 1e-12:
            best_phi = float(phi_grid[best_idx])
            best_side = float(sides_grid[best_idx])
            best_cx = float(center_final[0])
            best_cy = float(center_final[1])
    except Exception:
        pass
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
