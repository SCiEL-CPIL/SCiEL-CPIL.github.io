"""Combined program: honeycomb initial config + fast SAT + SA + Nelder-Mead polish."""
import numpy as np
import time
import random
import math
import scipy.optimize

N = 12
SQRT3 = math.sqrt(3.0)
DEADLINE = 145.0

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
    start_time = time.time()
    deadline = start_time + DEADLINE

    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS

    def get_all_vertices(positions, angles_rad):
        """Vectorized vertex computation for all hexagons."""
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        """
        Fast SAT check for two unit hexagons.
        h1, h2: (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy

        if dist_sq > 4.5:
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

        en = edge_normals
        en1x = en[:, 0] * c1 - en[:, 1] * s1
        en1y = en[:, 0] * s1 + en[:, 1] * c1
        en2x = en[:, 0] * c2 - en[:, 1] * s2
        en2y = en[:, 0] * s2 + en[:, 1] * c2

        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() - 1e-12 or p2.max() < p1.min() - 1e-12:
                return True

        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() - 1e-12 or p2.max() < p1.min() - 1e-12:
                return True

        return False

    def compute_min_side_fast(positions, angles_deg, center, angle_deg):
        """
        Compute minimum outer side length required to contain all inner hexagons.
        positions: (N, 2) array
        angles_deg: (N,) array in degrees
        center: (cx, cy)
        angle_deg: outer hexagon rotation in degrees
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        offsets = verts_flat - np.array(center)

        best_side = float('inf')
        # Scan outer angle
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg + angle_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / SQRT3) * max_projs
            s = np.max(side_vals)
            if s < best_side:
                best_side = s

        return best_side

    def initial_guess():
        """Create a valid initial configuration using honeycomb packing."""
        a1 = np.array([SQRT3, 0.0])
        a2 = np.array([SQRT3/2.0, 1.5])

        lattice_coords = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]

        positions = np.zeros((N, 2))
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2

        # Center the configuration
        centroid = np.mean(positions, axis=0)
        positions -= centroid

        angles = np.zeros(N)
        return positions, angles

    def objective(positions, angles):
        """Objective: minimize outer side length while ensuring disjointness."""
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles[i])
                h2 = (positions[j][0], positions[j][1], angles[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6

        center = np.mean(positions, axis=0)
        side = compute_min_side_fast(positions, angles, center, 0.0)
        return side

    best_positions, best_angles = initial_guess()
    best_score = objective(best_positions, best_angles)

    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_score = best_score

    # Simulated annealing
    T = 2.0
    T_min = 0.001
    cooling_rate = 0.9997
    step_size = 0.3
    max_steps = 30000

    step_count = 0
    while time.time() < deadline - 25.0 and step_count < max_steps:
        step_count += 1
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()

        n_perturb = random.randint(1, 2)
        for _ in range(n_perturb):
            idx = random.randint(0, N-1)
            coord = random.randint(0, 2)  # 0=x, 1=y, 2=angle
            current_step = step_size * (0.5 + 0.5 * T)
            if coord < 2:
                new_positions[idx, coord] += random.gauss(0, current_step)
            else:
                new_angles[idx] += random.gauss(0, current_step * 0.5)

        new_score = objective(new_positions, new_angles)

        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / max(T, T_min)):
            current_positions = new_positions
            current_angles = new_angles
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_positions = current_positions.copy()
                best_angles = current_angles.copy()

        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Polish with Nelder-Mead using smooth objective
    def smooth_penalty(positions):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
                if d < 2.0:
                    penalty += (2.0 - d) ** 2
        return penalty

    def flatten_params(positions, angles):
        params = np.zeros(3*N)
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = angles[i]
        return params

    def unflatten_params(params):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i][0] = params[3*i]
            positions[i][1] = params[3*i+1]
            angles[i] = params[3*i+2]
        return positions, angles

    def polish_objective(params):
        positions, angles = unflatten_params(params)
        center = np.mean(positions, axis=0)
        side = compute_min_side_fast(positions, angles, center, 0.0)
        pen = smooth_penalty(positions)
        return side + 20.0 * pen

    if time.time() < deadline - 10.0:
        try:
            init_params = flatten_params(best_positions, best_angles)
            result = scipy.optimize.minimize(
                polish_objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )

            p = result.x
            positions, angles = unflatten_params(p)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (positions[i][0], positions[i][1], angles[i])
                    h2 = (positions[j][0], positions[j][1], angles[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        valid = False
                        break
                if not valid: break

            if valid:
                center = np.mean(positions, axis=0)
                new_side = compute_min_side_fast(positions, angles, center, 0.0)
                if new_side < best_score:
                    best_score = new_side
                    best_positions = positions
                    best_angles = angles
        except Exception:
            pass

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), float(np.degrees(best_angles[i]))])

    center = (float(np.mean(best_positions[:, 0])), float(np.mean(best_positions[:, 1])))
    angle = 0.0
    side = compute_min_side_fast(best_positions, best_angles, center, angle)

    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            h1 = (best_positions[i][0], best_positions[i][1], best_angles[i])
            h2 = (best_positions[j][0], best_positions[j][1], best_angles[j])
            if not hexagons_disjoint_fast(h1, h2):
                valid = False
                break
        if not valid: break

    if not valid:
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
