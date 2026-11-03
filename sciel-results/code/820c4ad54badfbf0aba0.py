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

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)

    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

    # Support function of unit hexagon: max dot product with direction
    # For a regular hexagon with vertices at angles 0, 60, ..., 300 degrees
    # h(u) = max over vertices of v . u
    # This equals 1/sin(30°) * max over edge normals of |n . u| for unit hex
    # Actually, for unit hex (side=1), the support function in direction u is:
    # h(u) = (1/sin(π/6)) * max_i |n_i · u| where n_i are edge outward normals
    # Wait, let me think more carefully.
    # A regular hexagon with side 1 has circumradius 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # The support function h(u) = max_v v·u
    # For a regular hexagon, this can be computed as:
    # h(u) = max over the 6 vertices of (cos(θ_v - θ_u), sin(θ_v - θ_u)) · (1,0) rotated
    # Simpler: h(u) = max_i cos(θ_u - θ_i) where θ_i are vertex angles
    # But we need to account for rotation.

    def hex_support(directions, angle_rad):
        """Support function of rotated unit hexagon.
        directions: (M, 2) array of unit direction vectors
        angle_rad: rotation angle of hexagon
        Returns: (M,) array of support values
        """
        # Rotate directions by -angle to get them in hex's local frame
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        dx = directions[:, 0] * c + directions[:, 1] * s
        dy = -directions[:, 0] * s + directions[:, 1] * c
        # In local frame, hex has vertices at angles 0, 60, ..., 300
        # h_local(u) = max_i cos(angle(u) - i*60°)
        # = max over 6 values
        theta = np.arctan2(dy, dx)
        # h = max over k of cos(theta - k*pi/3)
        # This is equivalent to: h = 1/cos(pi/6) * max over edge normals of |n_k . u|
        # Actually simpler: h = max_k cos(theta - k*pi/3)
        # For efficiency, compute all 6
        k_angles = np.arange(6) * np.pi / 3
        cos_vals = np.cos(theta[None, :] - k_angles[:, None])
        return np.max(cos_vals, axis=0)

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

    def check_pair_sat(h1, h2):
        """Return True if hexagons h1, h2 are disjoint (separated)."""
        return hexagons_disjoint_fast(h1, h2)

    def check_moved_valid(positions, angles_rad, idx):
        """Check if hexagon idx is disjoint from all others."""
        h_moved = (positions[idx, 0], positions[idx, 1], angles_rad[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                return False
        return True

    def compute_min_outer_side(positions, angles_deg):
        """Compute minimum enclosing regular hexagon side length.
        Uses support functions for efficiency."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # For a regular hexagon with side s and rotation phi:
        # The constraint is that all points fit inside.
        # A point p is inside hex(center, s, phi) iff for all 6 edge normals n_k(phi):
        #   n_k · (p - center) <= s * h_edge_normal
        # where h_edge_normal is the support of unit hex in the edge normal direction.
        # For unit hex (side 1), the distance from center to each edge is 1/sin(30°) = 2... 
        # Wait, for unit hex with circumradius 1, inradius = cos(30°) = sqrt(3)/2
        # So the constraint is: |n_k · (p-center)| <= s * sqrt(3)/2 for all k
        # Side length s = 2 * max_k |n_k · (p-center)| / (sqrt(3)/2) = (4/sqrt(3)) * max_k |n_k · offset|
        # Hmm, let me re-derive.
        # Unit hex (side 1): circumradius R=1, inradius r=sqrt(3)/2
        # Scaled by s: circumradius s, inradius s*sqrt(3)/2
        # Edge normals at angles phi + 30° + k*60°
        # Constraint: |n_k · offset| <= s * sqrt(3)/2
        # So s >= 2*|n_k · offset| / sqrt(3) for all k
        # s_min = (2/sqrt(3)) * max_k max_p |n_k · offset_p|

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

        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)
        side_vals = (2.0 / sqrt3) * max_projs
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = np.max(side_vals[best_idx])
        best_phi = float(best_idx) * 0.5

        try:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-9})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def compute_min_outer_side_smooth(positions, angles_deg, beta=30.0):
        """Smooth approximation of the outer hexagon side length."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def eval_phi_smooth(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)

            # Soft max over vertices for each edge normal
            scaled = abs_proj / beta
            scaled -= scaled.max(axis=0, keepdims=True)
            log_sum_exp = beta * np.log(np.sum(np.exp(scaled), axis=0)) + scaled.max(axis=0, keepdims=True).squeeze(0)

            side_vals = (2.0 / sqrt3) * log_sum_exp
            # Soft max over edges
            scaled_side = side_vals / beta
            scaled_side -= scaled_side.max()
            soft_max_side = beta * np.log(np.sum(np.exp(scaled_side))) + scaled_side.max()
            return soft_max_side

        # Coarse scan
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)

        scaled = abs_proj / beta
        max_scaled = scaled.max(axis=1, keepdims=True)
        scaled -= max_scaled
        log_sum_exp_verts = beta * np.log(np.sum(np.exp(scaled), axis=1)) + max_scaled.squeeze(1)

        side_vals = (2.0 / sqrt3) * log_sum_exp_verts

        scaled_side = side_vals / beta
        max_side_scaled = scaled_side.max(axis=1, keepdims=True)
        scaled_side -= max_side_scaled
        soft_max_side = beta * np.log(np.sum(np.exp(scaled_side), axis=1)) + max_side_scaled.squeeze(1)

        best_idx = np.argmin(soft_max_side)
        best_side = soft_max_side[best_idx]
        best_phi = float(best_idx)

        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi_smooth, bounds=(lo, hi), method='bounded',
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

    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            a = k * np.pi / 3
            positions[k, 0] = 2*sqrt3 * math.cos(a)
            positions[k, 1] = 2*sqrt3 * math.sin(a)
        positions[6] = [0, 0]
        for k in range(5):
            a = k * 2*np.pi/5 + np.pi/5
            positions[7+k, 0] = 3.5*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3.5*sqrt3 * math.sin(a)
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
            positions[k+6, 0] = k * 2*sqrt3 + sqrt3/2
            positions[k+6, 1] = 3.0
        return positions, angles

    def make_compact_cluster():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        coords = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (2,-1), (-2,1), (1,1), (-1,-1), (2,0)]
        for idx, (i, j) in enumerate(coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_star_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            a = k * np.pi / 3
            positions[k, 0] = 1.8*sqrt3 * math.cos(a)
            positions[k, 1] = 1.8*sqrt3 * math.sin(a)
        for k in range(6):
            a = (k + 0.5) * np.pi / 3
            positions[6+k, 0] = 3.2*sqrt3 * math.cos(a)
            positions[6+k, 1] = 3.2*sqrt3 * math.sin(a)
        return positions, angles

    # === New structural approach: L-BFGS-B with smooth objective as primary ===

    def smooth_full_objective(params, beta=20.0):
        """Full smooth objective for L-BFGS-B.
        params: 3N array [x0, y0, a0, x1, y1, a1, ...]
        Returns smooth approximation of outer hex side + collision penalty."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]

        # Collision penalty (soft)
        angles_rad = np.radians(ang)
        collision_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Compute minimum penetration depth approximation
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Approximate overlap as (2*sqrt(3) - dist) for close hexagons
                    overlap = max(0, 2.0*sqrt3 - dist)
                    collision_pen += overlap * overlap * 10.0

        if collision_pen > 1e6:
            return 1e6

        # Smooth outer hex side
        side, _, _, _ = compute_min_outer_side_smooth(pos, ang, beta=beta)
        return side + collision_pen

    def smooth_full_gradient(params, beta=20.0):
        """Numerical gradient for L-BFGS-B."""
        eps = 1e-7
        grad = np.zeros(3*N)
        f0 = smooth_full_objective(params, beta)
        for i in range(3*N):
            params_p = params.copy()
            params_p[i] += eps
            fp = smooth_full_objective(params_p, beta)
            params_m = params.copy()
            params_m[i] -= eps
            fm = smooth_full_objective(params_m, beta)
            grad[i] = (fp - fm) / (2*eps)
        return grad

    def lbfgsb_optimize(pos_init, ang_init, time_limit, beta=20.0):
        """Run L-BFGS-B optimization with smooth objective."""
        t_start = time.time()
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = pos_init[i, 0]
            init_params[3*i+1] = pos_init[i, 1]
            init_params[3*i+2] = ang_init[i]

        try:
            res = minimize(smooth_full_objective, init_params, method='L-BFGS-B',
                           jac=smooth_full_gradient,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxfun': 50000})

            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]

                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    return new_side, new_pos, new_ang, new_phi, new_cx, new_cy
        except Exception:
            pass

        return None

    def sa_with_fast_eval(positions_init, angles_init, time_budget):
        """SA with fast evaluation using smooth objective for acceptance."""
        t_start = time.time()
        current_positions = positions_init.copy()
        current_angles = angles_init.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

        T = 0.5
        T_min = 0.0001
        cooling = 0.99997
        step_size = 0.3
        step_decay = 0.99999

        accept_count = 0
        total_count = 0
        stagnation_count = 0

        while time.time() - t_start < time_budget:
            total_count += 1

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
            if not check_moved_valid(new_positions, angles_rad_new, idx):
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.005: step_size = 0.005
                continue

            # Use smooth objective for quick evaluation
            new_side_smooth, _, _, _ = compute_min_outer_side_smooth(new_positions, new_angles, beta=30.0)

            if new_side_smooth < current_side - 0.01:
                # Accept and compute exact
                new_side_exact, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side_exact
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                stagnation_count = 0
                if new_side_exact < best_side:
                    best_side = new_side_exact
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
            else:
                delta = new_side_smooth - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    current_positions = new_positions
                    current_angles = new_angles
                    # Compute exact for tracking
                    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
                    accept_count += 1
                    if current_side < best_side:
                        best_side = current_side
                        best_positions = current_positions.copy()
                        best_angles = current_angles.copy()
                        best_phi = current_phi
                        best_cx = current_cx
                        best_cy = current_cy
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

            if total_count % 200 == 0:
                acc_rate = accept_count / total_count
                if acc_rate < 0.03:
                    step_size *= 0.8
                    if step_size < 0.005: step_size = 0.005
                elif acc_rate > 0.5:
                    step_size *= 1.2
                    if step_size > 1.5: step_size = 1.5
                accept_count = 0
                total_count = 0

                # If stuck, do a small perturbation
                if stagnation_count > 100:
                    for i in range(N):
                        current_positions[i, 0] += random.gauss(0, 0.2)
                        current_positions[i, 1] += random.gauss(0, 0.2)
                        current_angles[i] += random.gauss(0, 3.0)
                    angles_rad_check = np.radians(current_angles)
                    valid = True
                    for i in range(N):
                        if not check_moved_valid(current_positions, angles_rad_check, i):
                            valid = False
                            break
                    if valid:
                        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
                    else:
                        current_positions = best_positions.copy()
                        current_angles = best_angles.copy()
                        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
                    stagnation_count = 0

            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005

        return best_side, best_positions, best_angles, best_phi, best_cx, best_cy

    def polish_fine(pos, ang, time_limit):
        """Fine coordinate descent polish."""
        t_start = time.time()
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()

        step = 0.05
        min_step = 0.001

        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            for idx in range(N):
                if time.time() - t_start > time_limit:
                    break
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
                        if not check_moved_valid(new_pos, angles_rad_new, idx):
                            continue

                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)

                        if new_side < best_side_cd - 1e-12:
                            best_pos_cd = new_pos
                            best_ang_cd = new_ang
                            best_side_cd = new_side
                            improved = True
                            break

                    if improved:
                        break
                    if time.time() - t_start > time_limit:
                        break
                if improved:
                    break

            if not improved:
                step *= 0.7

        return best_side_cd, best_pos_cd, best_ang_cd

    # === Main optimization pipeline ===

    inits = [make_honeycomb(), make_ring_config(), make_two_rows(), make_compact_cluster(), make_star_config()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Phase 1: Evaluate all initial configurations
    for pos_init, ang_init in inits:
        s, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
        if s < best_side:
            best_side = s
            best_positions = pos_init.copy()
            best_angles = ang_init.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed

    # Phase 2: L-BFGS-B from best initial (structural change: gradient-based primary)
    if remaining > 20.0:
        lbfgs_time = min(remaining * 0.30, 35.0)
        result = lbfgsb_optimize(best_positions, best_angles, lbfgs_time, beta=20.0)
        if result is not None:
            s, pos, ang, phi, cx, cy = result
            if s < best_side:
                best_side = s
                best_positions = pos
                best_angles = ang
                best_phi = phi
                best_cx = cx
                best_cy = cy

    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed

    # Phase 3: SA from best (with fast smooth evaluation)
    if remaining > 15.0:
        sa_time = min(remaining * 0.45, 45.0)
        s, pos, ang, phi, cx, cy = sa_with_fast_eval(best_positions, best_angles, sa_time)
        if s < best_side:
            best_side = s
            best_positions = pos
            best_angles = ang
            best_phi = phi
            best_cx = cx
            best_cy = cy

    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed

    # Phase 4: L-BFGS-B again from SA result
    if remaining > 15.0:
        lbfgs_time = min(remaining * 0.40, 30.0)
        result = lbfgsb_optimize(best_positions, best_angles, lbfgs_time, beta=15.0)
        if result is not None:
            s, pos, ang, phi, cx, cy = result
            if s < best_side:
                best_side = s
                best_positions = pos
                best_angles = ang
                best_phi = phi
                best_cx = cx
                best_cy = cy

    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed

    # Phase 5: Fine coordinate descent polish
    if remaining > 5.0:
        cd_time = min(remaining * 0.50, 20.0)
        s, pos, ang = polish_fine(best_positions, best_angles, cd_time)
        if s < best_side:
            best_side = s
            best_positions = pos
            best_angles = ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed

    # Phase 6: Final L-BFGS-B polish with tight beta
    if remaining > 8.0:
        lbfgs_time = min(remaining * 0.60, 15.0)
        result = lbfgsb_optimize(best_positions, best_angles, lbfgs_time, beta=10.0)
        if result is not None:
            s, pos, ang, phi, cx, cy = result
            if s < best_side:
                best_side = s
                best_positions = pos
                best_angles = ang
                best_phi = phi
                best_cx = cx
                best_cy = cy

    # Final validation
    if not is_valid_configuration(best_positions, best_angles):
        for pos_f, ang_f in inits:
            if is_valid_configuration(pos_f, ang_f):
                best_positions = pos_f.copy()
                best_angles = ang_f.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
