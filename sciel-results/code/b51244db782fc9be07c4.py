import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill
SQRT3 = math.sqrt(3)

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Precompute outward normals for unit hexagon
    unit_hex_normals = np.zeros((6, 2))
    for i in range(6):
        ex = unit_hex_verts[(i+1)%6, 0] - unit_hex_verts[i, 0]
        ey = unit_hex_verts[(i+1)%6, 1] - unit_hex_verts[i, 1]
        unit_hex_normals[i, 0] = ey
        unit_hex_normals[i, 1] = -ex
        norm = math.sqrt(unit_hex_normals[i, 0]**2 + unit_hex_normals[i, 1]**2)
        if norm > 1e-10:
            unit_hex_normals[i, 0] /= norm
            unit_hex_normals[i, 1] /= norm
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq > 12.0:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        
        for i in range(6):
            nx = unit_hex_normals[i, 0] * c1 - unit_hex_normals[i, 1] * s1
            ny = unit_hex_normals[i, 0] * s1 + unit_hex_normals[i, 1] * c1
            
            proj1 = v1x * nx + v1y * ny
            proj2 = v2x * nx + v2y * ny
            
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
        
        for i in range(6):
            nx = unit_hex_normals[i, 0] * c2 - unit_hex_normals[i, 1] * s2
            ny = unit_hex_normals[i, 0] * s2 + unit_hex_normals[i, 1] * c2
            
            proj1 = v1x * nx + v1y * ny
            proj2 = v2x * nx + v2y * ny
            
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
                
        return False

    def compute_min_outer_side(positions, angles_deg):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        
        # Flatten to (72, 2)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center
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
            side_vals = (2.0 / SQRT3) * max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
                
        # Fine scan using minimize_scalar
        try:
            lo = max(0.0, best_phi - 2.0)
            hi = min(60.0, best_phi + 2.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        """Check if all hexagons are pairwise disjoint."""
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Initial Configuration: 1-5-6 Ring Structure
    # This topology allows for tighter packing around the center.
    # Ring 0: 1 hexagon at center
    # Ring 1: 5 hexagons surrounding it (pentagonal symmetry approximated by hex lattice)
    # Ring 2: 6 hexagons surrounding those
    
    # Actually, a standard hex lattice has 1-6-12... 
    # The inspiration mentions "1-5-6 ring structure". Let's interpret this as a specific arrangement.
    # Let's try a compact cluster.
    
    # A better initial guess for "1-5-6" might be:
    # Center: (0,0)
    # Inner Ring (5): Angles 0, 72, 144, 216, 288? No, hexagons don't fit perfectly in pentagons.
    # Let's stick to a valid hex lattice subset that is compact.
    # A 1-6-12 structure is standard.
    # Maybe the "1-5-6" refers to a specific non-lattice packing?
    # Let's use a standard dense hexagonal packing for initialization, which is known to be good.
    # Lattice vectors:
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3/2.0, 1.5])
    
    # 12 hexagons in a compact hexagonal shape (radius 2 in lattice coords)
    # Center (0,0)
    # Ring 1 (6): (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Ring 2 (5 needed to reach 12? No, 1+6+12=19. We need 12.
    # So 1 + 6 + 5 from the next ring?
    # Or just 1 + 11?
    # Let's pick 12 points from the hex lattice that form a compact blob.
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)  -> 7 total
    # Next ring candidates: (2,0), (-2,0), (0,2), (0,-2), (2,-1), (-2,1), (1,1), (-1,-1), (2,-2), (-2,2)
    # Let's pick 5 of these to make a roughly circular blob.
    # (2,0), (0,2), (-2,0), (0,-2), (1,1) ?
    
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (0, 2), (-2, 0), (0, -2), (2, -1)
    ]
    
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    # SA Loop
    T = 0.5
    T_min = 0.0001
    cooling = 0.9999
    step_size = 0.3
    step_decay = 0.99998
    
    step_count = 0
    max_steps = 200000
    sa_time_budget = DEADLINE * 0.65
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        
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

    # Polish using Nelder-Mead on the full parameter set
    # Parameters: cx, cy, phi, x0, y0, a0, ..., x11, y11, a11
    def polish_objective(params):
        cx, cy = params[0], params[1]
        phi_deg = params[2]
        phi = np.radians(phi_deg)
        
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3 + 3*i]
            pos[i, 1] = params[4 + 3*i]
            ang[i] = params[5 + 3*i]
        
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Compute required d_k for each of the six planes
        required_d = np.zeros(6)
        for k in range(6):
            normal_angle = phi + np.pi/6 + k * np.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            proj = (verts_flat[:, 0] - cx) * nx + (verts_flat[:, 1] - cy) * ny
            required_d[k] = np.max(proj)
        
        # Side length = max(2 * required_d[k] / SQRT3)
        # Use smooth max via log-sum-exp
        beta = 100.0
        d_safe = np.maximum(required_d, 1e-10)
        log_d = np.log(d_safe)
        log_sum = np.log(np.sum(np.exp(beta * log_d))) / beta
        smooth_side = (2.0 / SQRT3) * np.exp(log_sum)
        
        # Penalty for inner hexagon overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    penalty += 10.0
        
        return smooth_side + penalty

    init_params = np.zeros(3 + 3*N)
    init_params[0] = best_cx
    init_params[1] = best_cy
    init_params[2] = best_phi
    for i in range(N):
        init_params[3 + 3*i] = best_positions[i, 0]
        init_params[4 + 3*i] = best_positions[i, 1]
        init_params[5 + 3*i] = best_angles[i]
    
    # Run polish
    if time.time() - t0 < DEADLINE * 0.95:
        try:
            res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_side + 0.5:
                new_cx, new_cy = res.x[0], res.x[1]
                new_phi = res.x[2]
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3 + 3*i]
                    new_pos[i, 1] = res.x[4 + 3*i]
                    new_ang[i] = res.x[5 + 3*i]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi_opt, new_cx_opt, new_cy_opt = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi_opt
                        best_cx = new_cx_opt
                        best_cy = new_cy_opt
        except Exception:
            pass

    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
