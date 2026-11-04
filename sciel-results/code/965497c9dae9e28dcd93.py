import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
sqrt3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Unit hexagon vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = (unit_verts @ rot.T) * side
        verts += np.array([cx, cy])
        return verts

    def calc_outer_side(positions, angles, outer_angle_deg=0.0):
        """Exact minimum side length of regular hexagon centered at origin containing all inner hexagons."""
        theta = math.radians(outer_angle_deg)
        normals = np.array([[math.cos(theta + math.pi/6 + k * math.pi/3),
                             math.sin(theta + math.pi/6 + k * math.pi/3)] for k in range(6)])
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ normals.T
            mp = np.max(projs)
            if mp > max_proj:
                max_proj = mp
        return 2.0 * max_proj / sqrt3

    def sat_gap(poly1, poly2):
        """Maximum SAT separation. Positive = disjoint, negative = overlap."""
        max_gap = -float('inf')
        for poly in (poly1, poly2):
            for i in range(6):
                edge = poly[(i + 1) % 6] - poly[i]
                axis = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    continue
                axis /= norm
                p1 = poly1 @ axis
                p2 = poly2 @ axis
                if np.max(p1) < np.min(p2):
                    gap = np.min(p2) - np.max(p1)
                elif np.max(p2) < np.min(p1):
                    gap = np.min(p1) - np.max(p2)
                else:
                    gap = -(min(np.max(p1), np.max(p2)) - max(np.min(p1), np.min(p2)))
                if gap > max_gap:
                    max_gap = gap
        return max_gap

    def is_valid(positions, angles, tol=-1e-9):
        polys = [get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if sat_gap(polys[i], polys[j]) < tol:
                    return False
        return True

    # Generate honeycomb lattice points
    lattice = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = 1.5 * i
            y = sqrt3 / 2.0 * i + sqrt3 * j
            r2 = x * x + y * y
            if r2 < 25.0:  # limit radius
                lattice.append((x, y, r2))
    lattice.sort(key=lambda t: t[2])
    lattice_pts = np.array([(x, y) for x, y, _ in lattice])

    # Strategy: try multiple initial configurations and optimize
    best_side = float('inf')
    best_positions = None
    best_angles = None

    def try_config(init_pos, init_ang, maxiter=500):
        nonlocal best_side, best_positions, best_angles
        if time.time() > deadline - 10:
            return

        def objective(params):
            if time.time() > deadline - 5:
                return 1e6
            pos = params[:2*N].reshape(N, 2)
            ang = params[2*N:]
            
            # Outer side length
            side = calc_outer_side(pos, ang, 0.0)
            
            # Overlap penalty (hard constraint via penalty)
            polys = [get_hex_verts(pos[i, 0], pos[i, 1], 1.0, ang[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    g = sat_gap(polys[i], polys[j])
                    if g < 0:
                        penalty += 100.0 * g * g
            
            return side + penalty

        x0 = np.concatenate([init_pos.flatten(), init_ang])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 100.0:  # reasonable
                final_pos = res.x[:2*N].reshape(N, 2)
                final_ang = res.x[2*N:]
                if is_valid(final_pos, final_ang):
                    s = calc_outer_side(final_pos, final_ang, 0.0)
                    if s < best_side:
                        best_side = s
                        best_positions = final_pos.copy()
                        best_angles = final_ang.copy()
        except Exception:
            pass

    # Initial config 1: 12 closest honeycomb points centered
    init1 = lattice_pts[:12].copy()
    init1 -= np.mean(init1, axis=0)
    try_config(init1, np.zeros(N), 3000)

    # Initial config 2: 12 closest, but shift to break symmetry
    for shift in [0.1, -0.1, 0.2, -0.2]:
        if time.time() > deadline - 30:
            break
        init2 = lattice_pts[:12].copy()
        init2 -= np.mean(init2, axis=0)
        init2 += np.array([shift, 0])
        try_config(init2, np.zeros(N), 2000)

    # Initial config 3: random perturbations
    rng = np.random.default_rng(42)
    for restart in range(15):
        if time.time() > deadline - 20:
            break
        init3 = init1.copy() + rng.normal(0, 0.3, (N, 2))
        init_ang3 = rng.uniform(-30, 30, N)
        try_config(init3, init_ang3, 1500)

    # Try with small rotations on honeycomb
    if time.time() < deadline - 15:
        for rot_deg in [5, -5, 10, -10, 15]:
            if time.time() > deadline - 10:
                break
            init_rot = init1.copy()
            init_ang_rot = np.full(N, rot_deg)
            try_config(init_rot, init_ang_rot, 1000)

    # Final polish
    if best_positions is not None and time.time() < deadline - 8:
        def objective_polish(params):
            if time.time() > deadline - 3:
                return 1e6
            pos = params[:2*N].reshape(N, 2)
            ang = params[2*N:]
            side = calc_outer_side(pos, ang, 0.0)
            polys = [get_hex_verts(pos[i, 0], pos[i, 1], 1.0, ang[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    g = sat_gap(polys[i], polys[j])
                    if g < 0:
                        penalty += 1000.0 * g * g
            return side + penalty

        x0 = np.concatenate([best_positions.flatten(), best_angles])
        try:
            res = minimize(objective_polish, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            final_pos = res.x[:2*N].reshape(N, 2)
            final_ang = res.x[2*N:]
            if is_valid(final_pos, final_ang):
                s = calc_outer_side(final_pos, final_ang, 0.0)
                if s < best_side:
                    best_side = s
                    best_positions = final_pos.copy()
                    best_angles = final_ang.copy()
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_positions is None:
        best_positions = init1
        best_angles = np.zeros(N)
        best_side = calc_outer_side(best_positions, best_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    return inner, (0.0, 0.0), float(best_side), 0.0
