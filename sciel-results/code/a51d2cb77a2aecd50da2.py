import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def _soft_max(x, beta=50.0):
    """Smooth max via log-sum-exp."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _soft_min(x, beta=50.0):
    """Smooth min via negative log-sum-exp."""
    return -_soft_max(-x, beta)

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]
    
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
    
    def compute_min_outer_side(positions, angles_deg, fine_tune=True):
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
        
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        max_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(max_sides)
        best_side = float(max_sides[best_idx])
        best_phi = float(best_idx)
        
        if fine_tune:
            try:
                lo = max(0.0, best_phi - 1.0)
                hi = min(60.0, best_phi + 1.0)
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
            
        return best_side, best_phi, center[0], center[1]
    
    def compute_smooth_outer_side_and_grad(positions, angles_deg, beta=30.0):
        """
        Compute smooth surrogate for outer side length and its gradient w.r.t. positions.
        Uses log-sum-exp to make min/max differentiable.
        Returns (smooth_side, grad_positions [N,2], grad_angles [N])
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Coarse scan to find good phi
        phi_vals = np.radians(np.arange(0, 60, 2))
        best_phi = 0.0
        best_val = float('inf')
        for phi_deg in phi_vals:
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            val = _soft_max(side_vals, beta)
            if val < best_val:
                best_val = val
                best_phi = phi_deg
        
        # Fine tune phi
        lo = max(0.0, best_phi - 2.0)
        hi = min(60.0, best_phi + 2.0)
        try:
            def eval_smooth_phi(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return _soft_max(side_vals, beta)
            res = minimize_scalar(eval_smooth_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success:
                best_phi = res.x
                best_val = res.fun
        except Exception:
            pass
        
        # Compute gradients at best phi
        phi = np.radians(best_phi)
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # For each of 6 directions, compute projection and gradient
        # proj[i, k] = offset_i . n_k
        # side_k = (2/sqrt3) * max_i |proj[i,k]|
        # smooth_side = soft_max_k(side_k)
        
        # Gradient of soft_max: d(soft_max)/dx_k = exp(beta*(x_k - m)) / sum_j exp(beta*(x_j - m))
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)  # shape (6,)
        side_vals = (2.0 / sqrt3) * max_projs  # shape (6,)
        
        # Soft max weights
        m = np.max(side_vals)
        exp_terms = np.exp(beta * (side_vals - m))
        sum_exp = np.sum(exp_terms)
        sm_weights = exp_terms / sum_exp  # shape (6,)
        
        # Now compute gradient w.r.t. each vertex position
        # d(side_k)/d(offset_i) = (2/sqrt3) * sign(proj[i,k]) * n_k if |proj[i,k]| = max
        # In smooth version: use soft-max over vertices for each direction
        
        grad_pos = np.zeros((N * 6, 2))  # all vertices
        grad_angles = np.zeros(N * 6)
        
        for k in range(6):
            # Soft max over vertices for direction k
            abs_pk = abs_proj[:, k]  # shape (72,)
            m_v = np.max(abs_pk)
            exp_v = np.exp(beta * (abs_pk - m_v))
            sum_v = np.sum(exp_v)
            v_weights = exp_v / sum_v  # shape (72,)
            
            # d(side_k)/d(offset_i) = (2/sqrt3) * sm_weights[k] * v_weights[i] * sign(proj[i,k]) * n_k
            signs = np.sign(proj[:, k])
            factor = (2.0 / sqrt3) * sm_weights[k]
            for i in range(72):
                hex_idx = i // 6
                vert_idx = i % 6
                g = factor * v_weights[i] * signs[i] * np.array([nx[k], ny[k]])
                grad_pos[i, 0] += g[0]
                grad_pos[i, 1] += g[1]
        
        # Map vertex gradients to hexagon center gradients
        # Each hexagon has 6 vertices; gradient w.r.t. center is sum of vertex grads
        # Gradient w.r.t. angle: need rotation Jacobian
        grad_center = np.zeros((N, 2))
        grad_angle = np.zeros(N)
        
        for i in range(N):
            for v in range(6):
                idx = i * 6 + v
                grad_center[i] += grad_pos[idx]
            
            # Angle gradient: d(vert)/d(theta) = (-y_local, x_local) rotated
            # vert = R(theta) * local + center
            # d(vert)/d(theta) = R'(theta) * local = [-sin(theta)*lx - cos(theta)*ly, cos(theta)*lx - sin(theta)*ly]
            # Actually: d/dtheta [cos(t)*lx - sin(t)*ly, sin(t)*lx + cos(t)*ly]
            # = [-sin(t)*lx - cos(t)*ly, cos(t)*lx - sin(t)*ly]
            # = R(t + pi/2) * [lx, ly]
            
            theta = angles_rad[i]
            c_t, s_t = math.cos(theta), math.sin(theta)
            for v in range(6):
                idx = i * 6 + v
                lx = _UNIT_HEX_VERTS[v, 0]
                ly = _UNIT_HEX_VERTS[v, 1]
                # d(vert)/d(theta)
                dv_dx = -s_t * lx - c_t * ly
                dv_dy = c_t * lx - s_t * ly
                grad_angle[i] += grad_pos[idx, 0] * dv_dx + grad_pos[idx, 1] * dv_dy
        
        return best_val, grad_center, grad_angle
    
    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True
    
    # === Grid-based parametrization ===
    # 2x3 grid in hexagonal lattice: positions are i*a1 + j*a2
    # a1 = (sqrt3, 0), a2 = (sqrt3/2, 1.5) for unit hexagons
    # Grid: 2 rows (j=0,1) x 3 cols (i=0,1,2)
    # Or 3x2: 3 rows x 2 cols
    # Or 4x3: 4 rows x 3 cols (but that's 12)
    
    def make_grid_2x3(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        """2 rows, 3 cols hexagonal grid"""
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        # Apply anisotropic scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
    
    def make_grid_3x2(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        """3 rows, 2 cols hexagonal grid"""
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(3):
            for i in range(2):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
    
    def make_grid_4x3(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        """4 rows, 3 cols - but that's 12"""
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(4):
            for i in range(3):
                if idx >= N:
                    break
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
    
    def make_hex_ring(center_pos, radius, rotation_deg=0.0):
        """6 hexagons in a ring around center"""
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N + np.radians(rotation_deg)
            positions[i, 0] = center_pos[0] + radius * np.cos(theta)
            positions[i, 1] = center_pos[1] + radius * np.sin(theta)
            angles[i] = rotation_deg
        return positions, angles
    
    def make_two_clusters():
        """Two 2x3 grids side by side"""
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        # First 2x3 grid
        for j in range(2):
            for i in range(3):
                positions[idx] = i * a1 + j * a2
                angles[idx] = 0.0
                idx += 1
        # Second 2x3 grid offset
        offset = 3 * a1 + 0.5 * a2
        for j in range(2):
            for i in range(3):
                positions[idx] = (i) * a1 + j * a2 + offset
                angles[idx] = 0.0
                idx += 1
        return positions, angles
    
    # Generate initial configurations
    initial_configs = []
    
    # Grid 2x3 with various scales
    for s in [0.95, 0.98, 1.0, 1.02, 1.05]:
        initial_configs.append(make_grid_2x3(scale=s))
    
    # Grid 3x2
    for s in [0.95, 0.98, 1.0, 1.02, 1.05]:
        initial_configs.append(make_grid_3x2(scale=s))
    
    # Rotated grids
    for rot in [5, 10, 15, 20, 30]:
        initial_configs.append(make_grid_2x3(scale=1.0, rotation_deg=rot))
        initial_configs.append(make_grid_3x2(scale=1.0, rotation_deg=rot))
    
    # Anisotropic grids
    for sx in [0.95, 0.98, 1.0, 1.02, 1.05]:
        for sy in [0.95, 0.98, 1.0, 1.02, 1.05]:
            initial_configs.append(make_grid_2x3(scale=1.0, sx=sx, sy=sy))
    
    # Ring configurations
    for r in [2.0, 2.5, 3.0, 3.5]:
        initial_configs.append(make_hex_ring([0, 0], r))
    
    # Two clusters
    initial_configs.append(make_two_clusters())
    
    # Honeycomb
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    for scale in [0.9, 0.95, 1.0]:
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = (i * a1 + j * a2) * scale
            angles[idx] = 0.0
        initial_configs.append((positions, angles))
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Evaluate all initial configs
    for pos, ang in initial_configs:
        if time.time() - t0 > DEADLINE * 0.3:
            break
        try:
            side, phi, cx, cy = compute_min_outer_side(pos, ang, fine_tune=False)
            if is_valid_configuration(pos, ang) and side < best_side:
                best_side = side
                best_positions = pos.copy()
                best_angles = ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
        except Exception:
            continue
    
    if best_positions is None:
        best_positions, best_angles = make_grid_2x3(1.0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Projected gradient method with smooth surrogate (differentiable contact network)
    pg_deadline = t0 + DEADLINE * 0.45
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    # Projected gradient with active-set contact detection
    lr = 0.05
    lr_decay = 0.999
    beta_schedule = [50.0, 30.0, 20.0, 15.0, 10.0]
    current_beta = beta_schedule[0]
    beta_phase = 0
    stagnant = 0
    
    while time.time() < pg_deadline:
        # Compute smooth objective and gradient
        smooth_val, grad_pos, grad_ang = compute_smooth_outer_side_and_grad(
            current_positions, current_angles, beta=current_beta
        )
        
        # Take gradient step
        new_positions = current_positions - lr * grad_pos
        new_angles = current_angles - lr * grad_ang * (180.0 / math.pi)  # convert to degrees
        
        # Check validity and apply penalty if overlapping
        angles_rad_new = np.radians(new_angles)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (new_positions[i, 0], new_positions[i, 1], angles_rad_new[i])
                h2 = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Compute overlap penalty (approximate)
                    overlap_penalty += 0.1
                    # Push apart
                    dx = new_positions[i, 0] - new_positions[j, 0]
                    dy = new_positions[i, 1] - new_positions[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        dx, dy = 0.1, 0.1
                        dist = math.sqrt(0.02)
                    push = 0.05 / dist
                    new_positions[i, 0] += dx * push
                    new_positions[i, 1] += dy * push
                    new_positions[j, 0] -= dx * push
                    new_positions[j, 1] -= dy * push
        
        # Evaluate actual objective
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
        
        if new_side < current_side - 1e-10:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            stagnant = 0
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            stagnant += 1
        
        # Decay learning rate
        lr *= lr_decay
        if lr < 0.001:
            lr = 0.001
        
        # Adjust beta (smoothness) based on stagnation
        if stagnant > 50 and beta_phase < len(beta_schedule) - 1:
            beta_phase += 1
            current_beta = beta_schedule[beta_phase]
            lr = max(lr, 0.02)
            stagnant = 0
    
    # SA phase (shorter, for fine-tuning)
    sa_deadline = min(t0 + DEADLINE * 0.7, pg_deadline + DEADLINE * 0.25)
    
    T = 0.1
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.08
    step_decay = 0.99999
    
    while time.time() < sa_deadline:
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 2.0)
        
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
            if step_size < 0.003: step_size = 0.003
            continue
        
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
        
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
        if step_size < 0.003: step_size = 0.003
    
    # Coordinate descent polish
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 5.0:
        cd_deadline = t0 + remaining * 0.5
        step = 0.02
        min_step = 0.00005
        
        while time.time() < cd_deadline and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        
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
                        
                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
            else:
                step = max(step * 0.7, min_step)
        
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # Additional gradient-based polish
        remaining2 = DEADLINE - (time.time() - t0)
        if remaining2 > 3.0:
            pg2_deadline = t0 + remaining2 * 0.4
            lr2 = 0.01
            for _ in range(200):
                if time.time() > pg2_deadline:
                    break
                smooth_val, grad_pos, grad_ang = compute_smooth_outer_side_and_grad(
                    best_positions, best_angles, beta=100.0
                )
                new_pos = best_positions - lr2 * grad_pos
                new_ang = best_angles - lr2 * grad_ang * (180.0 / math.pi)
                
                angles_rad_new = np.radians(new_ang)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad_new[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                    if new_side < best_side - 1e-10:
                        best_positions = new_pos
                        best_angles = new_ang
                        best_side = new_side
                lr2 *= 0.98
    
    # Final fine-tune of outer hexagon angle
    remaining2 = DEADLINE - (time.time() - t0)
    if remaining2 > 3.0:
        def eval_final_phi(phi_deg):
            phi = np.radians(phi_deg)
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        try:
            res = minimize_scalar(eval_final_phi, bounds=(0, 60), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
                angles_rad = np.radians(best_angles)
                all_verts = get_all_vertices(best_positions, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                best_cx = center[0]
                best_cy = center[1]
        except Exception:
            pass
    
    # Validate and finalize
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to a known valid config
        best_positions, best_angles = make_grid_2x3(1.0)
    
    final_side, final_phi, final_cx, final_cy = compute_min_outer_side(best_positions, best_angles)
    if final_side < best_side:
        best_side = final_side
        best_phi = final_phi
        best_cx = final_cx
        best_cy = final_cy
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
