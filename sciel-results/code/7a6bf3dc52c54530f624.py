import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

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
    sqrt3 = math.sqrt(3)
    
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
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
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

    # Initial Configuration: A tight ring of 12 hexagons
    # This is a good starting point for 12 hexagons
    init_positions = np.zeros((N, 2))
    init_angles = np.zeros(N)
    
    # Place 12 hexagons in a circle around the center
    # Side length of outer hexagon for a ring of 12 unit hexagons is approx 4.5-5.0
    # We start with a radius that allows them to touch each other
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3)
    # For a ring of 12, the chord length is sqrt(3).
    # R * 2 * sin(pi/12) = sqrt(3) => R = sqrt(3) / (2 * sin(pi/12))
    R_ring = math.sqrt(3) / (2 * math.sin(math.pi / 12))
    
    for i in range(N):
        angle = i * 2 * math.pi / N
        init_positions[i, 0] = R_ring * math.cos(angle)
        init_positions[i, 1] = R_ring * math.sin(angle)
        # Orient hexagons to face outward? Or just 0? 
        # Let's try aligning them with the radial direction for a "flower" pattern
        # Actually, standard honeycomb is better. Let's use a mix.
        # Let's stick to a simple ring first, let the optimizer rotate them.
        init_angles[i] = 0.0

    best_side = float('inf')
    best_positions = init_positions.copy()
    best_angles = init_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate initial
    cur_side, cur_phi, cur_cx, cur_cy = compute_min_outer_side(init_positions, init_angles)
    if cur_side < best_side:
        best_side = cur_side
        best_positions = init_positions.copy()
        best_angles = init_angles.copy()
        best_phi = cur_phi
        best_cx = cur_cx
        best_cy = cur_cy

    # Strategy: Differentiable Optimization with Soft Constraints
    # Objective: Minimize (Outer Side Length) + Lambda * (Sum of Overlap Penalties)
    # We use a smooth approximation of the collision constraint.
    
    def objective_and_grad(params):
        # params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        pos = params[0:3*N:3]
        pos_y = params[1:3*N:3]
        angles_deg = params[2:3*N:3]
        
        positions = np.column_stack((pos, pos_y))
        angles_rad = np.radians(angles_deg)
        
        # 1. Compute Outer Side Length
        side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
        
        # 2. Compute Overlap Penalty
        # For each pair, compute the minimum distance between their boundaries.
        # If distance < 0, they overlap.
        # We use a soft penalty: sum(max(0, -dist)^2)
        # To make it differentiable, we use a smooth approximation or just rely on the fact 
        # that L-BFGS can handle non-smoothness somewhat, but better to be smooth.
        # A common trick is to use the support function difference.
        
        penalty = 0.0
        # Approximate overlap by checking distance between centers vs sum of radii?
        # No, hexagons are not circles.
        # Let's use a simpler proxy: if they are close, add a repulsive force.
        # Distance between centers d. If d < 2*inradius (sqrt(3)), they *might* overlap.
        # But this is not accurate.
        
        # Better: Use the exact disjoint check but penalize "nearness".
        # Since we want to use gradients, let's define a smooth repulsion.
        # For each pair (i, j):
        #   dx = x_i - x_j, dy = y_i - y_j
        #   dist = sqrt(dx^2 + dy^2)
        #   If dist < threshold (e.g. 1.8), apply penalty.
        #   Penalty = (threshold - dist)^2
        # This is differentiable everywhere except dist=0, which we avoid.
        
        # However, this doesn't account for rotation.
        # A rotation-aware soft constraint is complex.
        # Alternative: Just use the side length minimization with a hard constraint check 
        # inside the objective? No, that breaks gradients.
        
        # Let's try a different approach: 
        # Minimize side length, but if invalid, add a large penalty based on "depth of intrusion".
        # We can estimate intrusion depth by the amount by which the separating axis theorem fails.
        # This is hard to vectorize smoothly.
        
        # Let's stick to a robust heuristic: 
        # The current SA is good at finding valid configs. 
        # The bottleneck is polishing.
        # Let's use L-BFGS on a SMOOTH surrogate for the side length, 
        # and use the exact validity check to reject steps that become invalid.
        
        # Actually, let's just use the smooth log-sum-exp approximation for the max projection 
        # in the side length calculation, which makes the objective smooth.
        # And for collisions, we will use a "soft" repulsion based on center distance 
        # as a first-order approximation, refined by the exact check.
        
        # Let's refine the side length calculation to be smooth.
        # Instead of max, use log-sum-exp with a small beta.
        
        angles_rad_for_calc = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad_for_calc)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Smooth max for side length
        # side(phi) = max_i |proj_i| * 2/sqrt(3)
        # We optimize over phi as well? No, phi is determined by the configuration.
        # Wait, the current compute_min_outer_side finds the best phi.
        # To make this differentiable, we need to include phi in the parameters 
        # or use an envelope theorem approach.
        # For simplicity, let's keep phi fixed to 0 during optimization 
        # and rotate the whole system at the end? 
        # No, the relative orientations matter.
        
        # Let's just minimize the side length for a FIXED phi=0 (or best initial phi) 
        # and then rotate the whole configuration to find the global min side.
        # This decouples the "shape" optimization from the "orientation" optimization.
        
        # Objective: Minimize side length of the bounding hexagon (aligned with axes)
        # + Penalty for overlaps.
        
        # Side length of axis-aligned bounding hexagon:
        # Normals at 30, 90, 150, 210, 270, 330 degrees.
        phi_fixed = 0.0
        normal_angles = phi_fixed + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        
        # Smooth max
        beta = 10.0
        log_sum_exp = np.log(np.sum(np.exp(beta * abs_proj), axis=0) / 6.0) / beta
        max_projs_smooth = log_sum_exp
        side_smooth = (2.0 / sqrt3) * np.max(max_projs_smooth)
        
        # Overlap Penalty
        # Use a simple distance-based repulsion to keep them apart
        # This is a proxy, not exact, but helps the gradient descent avoid obvious collisions.
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i] - pos[j]
                dy = pos_y[i] - pos_y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                # If dist < 1.8, they are likely overlapping or touching
                # Soft repulsion
                if dist < 1.8:
                    # Penalty grows as they get closer
                    penalty += (1.8 - dist)**2
                    
        return side_smooth + 10.0 * penalty

    # We will run L-BFGS-B on this smooth objective.
    # We need to ensure the result is valid.
    
    current_params = np.zeros(3*N)
    for i in range(N):
        current_params[3*i] = best_positions[i, 0]
        current_params[3*i+1] = best_positions[i, 1]
        current_params[3*i+2] = best_angles[i]
        
    # Run minimization
    try:
        # Use a time limit for the optimizer
        res = minimize(objective_and_grad, current_params, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < objective_and_grad(current_params):
            new_params = res.x
            new_pos = np.column_stack((new_params[0:3*N:3], new_params[1:3*N:3]))
            new_ang = new_params[2:3*N:3]
            
            # Check validity
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

    # If the L-BFGS result was invalid or didn't improve, try a few restarts with different initializations
    # Or just use the SA result if it was better.
    # The current code had SA. Let's keep a small SA phase to ensure we have a valid config to start with.
    
    # Let's actually implement a quick SA to find a good valid config, then polish with L-BFGS.
    # The previous code did SA for 65% of the time. Let's do SA for 50%, then L-BFGS.
    
    # Re-initialize for SA
    t_sa_start = time.time()
    sa_budget = 60.0 # 60 seconds for SA
    
    # Use the ring initialization
    current_positions = init_positions.copy()
    current_angles = init_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0001
    cooling = 0.9999
    step_size = 0.3
    step_decay = 0.99998
    
    while time.time() - t_sa_start < sa_budget:
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 5.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
            continue
            
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
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
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # Now Polish with L-BFGS using the smooth objective
    if time.time() - t0 < DEADLINE - 5.0:
        current_params = np.zeros(3*N)
        for i in range(N):
            current_params[3*i] = best_positions[i, 0]
            current_params[3*i+1] = best_positions[i, 1]
            current_params[3*i+2] = best_angles[i]
            
        try:
            res = minimize(objective_and_grad, current_params, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < objective_and_grad(current_params):
                new_params = res.x
                new_pos = np.column_stack((new_params[0:3*N:3], new_params[1:3*N:3]))
                new_ang = new_params[2:3*N:3]
                
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

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to initial ring if it was valid
        if is_valid_configuration(init_positions, init_angles):
            best_positions = init_positions.copy()
            best_angles = init_angles.copy()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        else:
            # Should not happen, but just in case
            best_positions = np.zeros((N, 2))
            best_angles = np.zeros(N)
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
