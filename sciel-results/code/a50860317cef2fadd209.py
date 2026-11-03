import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

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
    
    sqrt3 = math.sqrt(3)
    
    # Precompute for speed
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

    def check_collision(h1, h2):
        """
        Returns True if h1 and h2 overlap.
        h1, h2 are (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        # Max distance between centers for touching hexagons is 2*sqrt(3)
        if dist_sq > 12.0: # (2*sqrt(3))^2 = 12
            return False
        
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Vertices of h1
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        # Vertices of h2
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Normals of h1
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        # Normals of h2
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # Separation axis theorem
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        return True

    def _eval_phi_fast(offsets, phi_rad, sqrt3):
        """Fast side-length eval for a single phi angle (radians)."""
        na = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(na)
        ny = np.sin(na)
        proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
        max_projs = np.max(np.abs(proj), axis=0)
        return (2.0 / sqrt3) * np.max(max_projs)

    def compute_outer_side_and_center(positions, angles_deg, refine=True):
        """
        Computes the minimum side length of the outer hexagon containing all inner hexagons.
        Returns (side_length, center_x, center_y, optimal_rotation_deg)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Compute centroid
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        if not refine:
            # Fast path: just evaluate at phi=0
            side = _eval_phi_fast(offsets, 0.0, sqrt3)
            return side, center[0], center[1], 0.0
        
        # Fully vectorized: compute side length for all 60 phi angles at once
        phi_vals = np.arange(0, 60, 1.0)
        phi_rad = np.radians(phi_vals)
        
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(best_sides)
        best_side = best_sides[best_idx]
        best_phi = phi_vals[best_idx]
        
        # Refine the best angle using bounded minimize
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            return _eval_phi_fast(offsets, phi, sqrt3)
            
        try:
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, center[0], center[1], best_phi

    def collision_penalty_smooth(pos, ang, beta=50.0):
        """
        Smooth log-sum-exp penalty for collisions.
        Uses the minimum gap across all 12 SAT axes per pair.
        """
        angles_rad = np.radians(ang)
        total = 0.0
        for i in range(N):
            for j in range(i+1, N):
                c1, s1 = math.cos(angles_rad[i]), math.sin(angles_rad[i])
                c2, s2 = math.cos(angles_rad[j]), math.sin(angles_rad[j])
                
                v1x = pos[i, 0] + _ux * c1 - _uy * s1
                v1y = pos[i, 1] + _ux * s1 + _uy * c1
                v2x = pos[j, 0] + _ux * c2 - _uy * s2
                v2y = pos[j, 1] + _ux * s2 + _uy * c2
                
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                # Collect all 12 gap values
                gaps = np.empty(12)
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gaps[k] = p2.min() - p1.max()
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gaps[6+k] = p2.min() - p1.max()
                
                # Log-sum-exp for soft-min of gaps (negative = overlap)
                # We want penalty when min_gap < 0
                # soft_min = -logsumexp(-gaps) / beta ... actually use:
                # penalty = log(sum(exp(-beta * gap)) for gap in gaps) / beta
                # This is smooth and ~ max(-gap) when gaps are close
                m = np.max(-gaps)  # for numerical stability
                if m > 100:
                    m = 100
                lse = m + math.log(np.sum(np.exp(-gaps - m)))
                total += lse / beta
        
        return total

    def is_valid(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if check_collision(h1, h2):
                    return False
        return True

    # --- Initialization: Honeycomb ---
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_cx = 0.0
    best_cy = 0.0
    best_phi = 0.0

    # --- Phase 1: Simulated Annealing ---
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_cx, current_cy, current_phi = compute_outer_side_and_center(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_cx = current_cx
        best_cy = current_cy
        best_phi = current_phi

    T = 0.5
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.3
    step_decay = 0.99998
    
    sa_time_budget = DEADLINE * 0.45
    max_steps = 100000
    step_count = 0
    
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
            if check_collision(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
            continue
            
        new_side, new_cx, new_cy, new_phi = compute_outer_side_and_center(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_cx = new_cx
            current_cy = new_cy
            current_phi = new_phi
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_cx = new_cx
                best_cy = new_cy
                best_phi = new_phi
        else:
            delta = new_side - current_side
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_cx = new_cx
                current_cy = new_cy
                current_phi = new_phi
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_cx = new_cx
                    best_cy = new_cy
                    best_phi = new_phi
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # --- Phase 2: Smooth Surrogate Optimization ---
    # Use L-BFGS-B on a smooth objective that combines side length and collision penalty
    def smooth_objective(x, beta=50.0):
        """
        Smooth objective: side_length + smooth collision_penalty
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = x[i*3]
            pos[i, 1] = x[i*3+1]
            ang[i] = x[i*3+2]
        
        # Compute side length (no refine for speed during optimization)
        side, _, _, _ = compute_outer_side_and_center(pos, ang, refine=False)
        
        # Compute smooth collision penalty
        penalty = collision_penalty_smooth(pos, ang, beta)
        
        return side + penalty
    
    def polish_smooth_lbfgs(pos, ang, time_limit):
        """
        Polish using L-BFGS-B on smooth surrogate with log-sum-exp collision penalty
        """
        n_params = N * 3
        x0 = np.zeros(n_params)
        for i in range(N):
            x0[i*3] = pos[i, 0]
            x0[i*3+1] = pos[i, 1]
            x0[i*3+2] = ang[i]
        
        t_start = time.time()
        
        best_side = float('inf')
        best_pos = pos.copy()
        best_ang = ang.copy()
        
        # Try multiple beta values, increasing for tighter constraint
        for beta in [30.0, 60.0, 120.0]:
            elapsed = time.time() - t_start
            if elapsed > time_limit * 0.7:
                break
            
            def obj(x):
                return smooth_objective(x, beta)
            
            try:
                res = minimize(
                    obj, x0, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
                )
                
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[i*3]
                    new_pos[i, 1] = res.x[i*3+1]
                    new_ang[i] = res.x[i*3+2]
                
                # Verify validity
                angles_rad = np.radians(new_ang)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                        if check_collision(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                    if side < best_side:
                        best_side = side
                        best_pos = new_pos.copy()
                        best_ang = new_ang.copy()
                        # Use this as starting point for next beta
                        x0 = res.x.copy()
            except Exception:
                pass
        
        # Final: use best valid result
        if best_side == float('inf'):
            best_side, _, _, _ = compute_outer_side_and_center(pos, ang)
        
        return best_side, best_pos, best_ang

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        # Phase 2: Smooth L-BFGS-B polish
        lbfgs_side, lbfgs_pos, lbfgs_ang = polish_smooth_lbfgs(best_positions, best_angles, remaining_time * 0.5)
        if lbfgs_side < best_side:
            best_side = lbfgs_side
            best_positions = lbfgs_pos
            best_angles = lbfgs_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
        
    # Phase 3: Coordinate Descent final polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 3.0:
            def polish_coordinate_descent(pos, ang, time_limit):
                """Coordinate descent with golden-section line search per parameter."""
                current_side, _, _, _ = compute_outer_side_and_center(pos, ang)
                best_side_cd = current_side
                best_pos_cd = pos.copy()
                best_ang_cd = ang.copy()
                
                t_start = time.time()
                
                # Try decreasing step sizes
                step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
                
                for step in step_sizes:
                    if time.time() - t_start > time_limit * 0.95:
                        break
                    
                    improved = False
                    for idx in range(N):
                        if time.time() - t_start > time_limit * 0.95:
                            break
                        
                        for dim in range(3):
                            if time.time() - t_start > time_limit * 0.95:
                                break
                            
                            # Try positive direction
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
                                    if check_collision(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                                
                                if new_side < best_side_cd - 1e-10:
                                    best_pos_cd = new_pos
                                    best_ang_cd = new_ang
                                    best_side_cd = new_side
                                    improved = True
                                    break
                            
                            if time.time() - t_start > time_limit * 0.95:
                                break
                        if time.time() - t_start > time_limit * 0.95:
                            break
                    
                    if not improved:
                        continue
                
                return best_side_cd, best_pos_cd, best_ang_cd
            
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.9)
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # Final verification
    if not is_valid(best_positions, best_angles):
        # Fallback to base if something went wrong
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    # Ensure we have a valid final computation with refinement
    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# Add differential evolution for global search improvement
def _global_search_init():
    """
    Generate diverse initial configurations for global search
    """
    configs = []
    
    # Honeycomb
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos[idx] = i * a1 + j * a2
    configs.append((pos.copy(), ang.copy()))
    
    # Ring configuration
    pos_ring = np.zeros((N, 2))
    ang_ring = np.zeros(N)
    for i in range(N):
        theta = 2 * math.pi * i / N
        r = 2.5
        pos_ring[i, 0] = r * math.cos(theta)
        pos_ring[i, 1] = r * math.sin(theta)
        ang_ring[i] = 0.0
    configs.append((pos_ring.copy(), ang_ring.copy()))
    
    # Random with some structure
    for _ in range(3):
        pos_rand = np.zeros((N, 2))
        ang_rand = np.zeros(N)
        # Place in rough hexagonal pattern with random perturbations
        for i in range(N):
            angle = 2 * math.pi * i / N
            r = 2.0 + random.gauss(0, 0.5)
            pos_rand[i, 0] = r * math.cos(angle) + random.gauss(0, 0.3)
            pos_rand[i, 1] = r * math.sin(angle) + random.gauss(0, 0.3)
            ang_rand[i] = random.gauss(0, 10.0)
        configs.append((pos_rand.copy(), ang_rand.copy()))
    
    return configs
