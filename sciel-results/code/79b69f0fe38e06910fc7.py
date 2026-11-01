"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill


def logsumexp(a, b=1.0):
    """Stable log-sum-exp for array a with temperature b."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a)
    if not np.isfinite(amax):
        return amax
    return amax + math.log(np.sum(np.exp((a - amax) / b))) * b


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def construct():
    start_time = time.time()

    # Initial layout: 11 hexagons in a compact hexagonal cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2),
    ]

    scale = 1.0 + 1e-6

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Phase 1: Simulated Annealing with direct containment objective
    # Key structural change: instead of optimizing outer params for fixed inner,
    # we directly minimize the required side length via SA moves on inner positions/rotations
    # AND outer center/angle simultaneously.

    def sa_objective(params):
        """Direct objective: compute required side length + overlap penalty."""
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)

        # Overlap penalty using distance-based approximation (fast)
        penalty = 0.0
        min_dist = SQRT3 - 0.01
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2

        return side + 50.0 * penalty

    def sa_validate(params):
        """Check if solution is valid (no overlaps, all inside)."""
        inner_list = []
        for i in range(N):
            inner_list.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)

        if not check_all_disjoint(inner_list):
            return False, None, None, None

        if not check_inside_outer(inner_list, (cx, cy), side, angle):
            return False, None, None, None

        return True, inner_list, (cx, cy), angle

    # Initialize SA state
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    init_params = []
    for x, y, a in initial_inner:
        init_params.extend([x, y, a])
    init_params.extend([cx_avg, cy_avg, 0.0])

    current_params = init_params[:]
    current_obj = sa_objective(current_params)

    best_sa_params = current_params[:]
    best_sa_obj = current_obj
    best_sa_valid = False
    best_sa_inner = None
    best_sa_center = None
    best_sa_angle = None

    # Check initial validity
    valid, inner_l, center_l, angle_l = sa_validate(current_params)
    if valid:
        best_sa_valid = True
        best_sa_inner = inner_l
        best_sa_center = center_l
        best_sa_angle = angle_l
        best_sa_obj = compute_required_side(inner_l, center_l, angle_l)

    # SA parameters
    T0 = 0.5
    T_end = 0.001
    total_steps = 50000
    steps_per_check = 500

    np.random.seed(42)

    for step in range(total_steps):
        if time.time() - start_time > DEADLINE - 30:
            break

        # Cool temperature
        T = T0 * (T_end / T0) ** (step / total_steps)

        # Generate a random perturbation
        new_params = current_params[:]
        move_type = np.random.randint(0, 5)

        if move_type < 3:
            # Perturb one inner hexagon's position
            idx = np.random.randint(0, N)
            base = idx * 3
            new_params[base] += np.random.uniform(-0.1, 0.1)
            new_params[base + 1] += np.random.uniform(-0.1, 0.1)
        elif move_type == 3:
            # Perturb one inner hexagon's rotation
            idx = np.random.randint(0, N)
            new_params[idx * 3 + 2] += np.random.uniform(-5, 5)
        else:
            # Perturb outer center or angle
            if np.random.random() < 0.7:
                new_params[N * 3] += np.random.uniform(-0.1, 0.1)
                new_params[N * 3 + 1] += np.random.uniform(-0.1, 0.1)
            else:
                new_params[N * 3 + 2] += np.random.uniform(-5, 5)

        new_obj = sa_objective(new_params)

        # Accept or reject
        if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / max(T, 1e-10)):
            current_params = new_params
            current_obj = new_obj

            # Track best valid solution
            if step % steps_per_check == 0:
                valid, inner_l, center_l, angle_l = sa_validate(current_params)
                if valid:
                    sa_side = compute_required_side(inner_l, center_l, angle_l)
                    if sa_side < best_sa_obj:
                        best_sa_valid = True
                        best_sa_inner = inner_l
                        best_sa_center = center_l
                        best_sa_angle = angle_l
                        best_sa_obj = sa_side
        else:
            # Rejected: revert
            pass

    # Update global best from SA
    if best_sa_valid and best_sa_obj < best_side:
        best_side = best_sa_obj
        best_inner = best_sa_inner
        best_outer_center = best_sa_center
        best_outer_angle = best_sa_angle

    # Phase 2: Local polish with Nelder-Mead on best SA solution
    if time.time() - start_time < DEADLINE - 20:
        try:
            if best_sa_valid:
                polish_params = best_sa_inner[:] + [best_outer_center[0], best_outer_center[1], best_outer_angle]
                # Flatten for optimizer
                flat = []
                for i, j, a in best_sa_inner:
                    flat.extend([i, j, a])
                flat.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

                def polish_obj(p):
                    inner_l = []
                    for i in range(N):
                        inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                    cx = p[N*3]
                    cy = p[N*3+1]
                    ang = p[N*3+2]
                    side = compute_required_side(inner_l, (cx, cy), ang)
                    # Penalty for overlaps
                    pen = 0.0
                    min_d = SQRT3 - 0.005
                    for i in range(N):
                        for j in range(i+1, N):
                            d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                            if d < min_d:
                                pen += (min_d - d)**2
                    return side + 100.0 * pen

                polish_res = minimize(
                    polish_obj,
                    flat,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000}
                )

                inner_l = []
                for i in range(N):
                    inner_l.append([polish_res.x[i*3], polish_res.x[i*3+1], polish_res.x[i*3+2]])
                cx = polish_res.x[N*3]
                cy = polish_res.x[N*3+1]
                ang = polish_res.x[N*3+2]

                side = compute_required_side(inner_l, (cx, cy), ang)

                if check_all_disjoint(inner_l) and check_inside_outer(inner_l, (cx, cy), side, ang):
                    if side < best_side:
                        best_side = side
                        best_inner = inner_l
                        best_outer_center = (cx, cy)
                        best_outer_angle = ang
        except Exception:
            pass

    # Phase 3: Multi-start with different initial topologies
    if time.time() - start_time < DEADLINE - 15:
        # Try a different initial arrangement: ring + center
        ring_coords = [(0, 0)]
        for i in range(6):
            q = [1, 0, -1, -1, 0, 1][i]
            r = [0, 1, 1, 0, -1, -1][i]
            ring_coords.append((q, r))
        # Add 4 more in outer ring
        ring_coords.append((2, 0))
        ring_coords.append((1, 1))
        ring_coords.append((0, 2))
        ring_coords.append((-1, 2))

        ring_centers = []
        for q, r in ring_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            ring_centers.append((x, y))

        ring_inner = [[x, y, 0.0] for x, y in ring_centers]

        # Quick SA on this topology
        cx_r = sum(c[0] for c in ring_centers) / len(ring_centers)
        cy_r = sum(c[1] for c in ring_centers) / len(ring_centers)

        init_p2 = []
        for x, y, a in ring_inner:
            init_p2.extend([x, y, a])
        init_p2.extend([cx_r, cy_r, 0.0])

        cur_p2 = init_p2[:]
        cur_obj2 = sa_objective(cur_p2)
        best_p2_params = cur_p2[:]
        best_p2_obj = cur_obj2
        best_p2_valid = False
        best_p2_inner = None
        best_p2_center = None
        best_p2_angle = None

        valid2, il2, cl2, al2 = sa_validate(cur_p2)
        if valid2:
            best_p2_valid = True
            best_p2_inner = il2
            best_p2_center = cl2
            best_p2_angle = al2
            best_p2_obj = compute_required_side(il2, cl2, al2)

        T0_2 = 0.4
        T_end_2 = 0.002
        steps2 = 20000

        np.random.seed(123)

        for step2 in range(steps2):
            if time.time() - start_time > DEADLINE - 15:
                break

            T2 = T0_2 * (T_end_2 / T0_2) ** (step2 / steps2)

            new_p2 = cur_p2[:]
            mt = np.random.randint(0, 4)
            if mt < 2:
                idx = np.random.randint(0, N)
                base = idx * 3
                new_p2[base] += np.random.uniform(-0.1, 0.1)
                new_p2[base + 1] += np.random.uniform(-0.1, 0.1)
            elif mt == 2:
                idx = np.random.randint(0, N)
                new_p2[idx * 3 + 2] += np.random.uniform(-5, 5)
            else:
                new_p2[N * 3] += np.random.uniform(-0.1, 0.1)
                new_p2[N * 3 + 1] += np.random.uniform(-0.1, 0.1)

            new_obj2 = sa_objective(new_p2)

            if new_obj2 < cur_obj2 or np.random.random() < math.exp(-(new_obj2 - cur_obj2) / max(T2, 1e-10)):
                cur_p2 = new_p2
                cur_obj2 = new_obj2

                if step2 % 500 == 0:
                    valid2, il2, cl2, al2 = sa_validate(cur_p2)
                    if valid2:
                        s2 = compute_required_side(il2, cl2, al2)
                        if s2 < best_p2_obj:
                            best_p2_valid = True
                            best_p2_inner = il2
                            best_p2_center = cl2
                            best_p2_angle = al2
                            best_p2_obj = s2

        if best_p2_valid and best_p2_obj < best_side:
            best_side = best_p2_obj
            best_inner = best_p2_inner
            best_outer_center = best_p2_center
            best_outer_angle = best_p2_angle

    # Phase 4: Final fine polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            flat_final = []
            for i, j, a in best_inner:
                flat_final.extend([i, j, a])
            flat_final.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

            def final_obj(p):
                inner_l = []
                for i in range(N):
                    inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                cx = p[N*3]
                cy = p[N*3+1]
                ang = p[N*3+2]
                side = compute_required_side(inner_l, (cx, cy), ang)
                pen = 0.0
                min_d = SQRT3 - 0.003
                for i in range(N):
                    for j in range(i+1, N):
                        d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                        if d < min_d:
                            pen += (min_d - d)**2
                return side + 200.0 * pen

            final_res = minimize(
                final_obj,
                flat_final,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 8000}
            )

            inner_l = []
            for i in range(N):
                inner_l.append([final_res.x[i*3], final_res.x[i*3+1], final_res.x[i*3+2]])
            cx = final_res.x[N*3]
            cy = final_res.x[N*3+1]
            ang = final_res.x[N*3+2]

            side = compute_required_side(inner_l, (cx, cy), ang)

            if check_all_disjoint(inner_l) and check_inside_outer(inner_l, (cx, cy), side, ang):
                if side < best_side:
                    best_side = side
                    best_inner = inner_l
                    best_outer_center = (cx, cy)
                    best_outer_angle = ang
        except Exception:
            pass

    # Final verification and fallback
    final_inner = [list(x) for x in best_inner]
    final_center = best_outer_center
    final_side = best_side
    final_angle = best_outer_angle

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        final_inner = [[x, y, 0.0] for x, y in fallback_centers]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_required_side(final_inner, final_center, final_angle)

    return final_inner, final_center, final_side, final_angle
