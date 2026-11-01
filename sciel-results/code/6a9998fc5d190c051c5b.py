"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 135.0  # safety margin before 150s kill


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


def compute_required_side_smooth(inner, outer_center, outer_angle_deg, temp=0.05):
    """
    Compute a smooth approximation of the minimum outer side length using log-sum-exp.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    all_signed = []

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                all_signed.append(proj)
                all_signed.append(-proj)

    if not all_signed:
        return 0.0

    all_signed = np.array(all_signed, dtype=np.float64)
    amax = np.max(all_signed)
    if not np.isfinite(amax):
        return float('inf')
    max_abs_proj_smooth = amax + math.log(np.sum(np.exp((all_signed - amax) / temp))) * temp

    return 2.0 * max_abs_proj_smooth / SQRT3


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


def fast_overlap_check(inner_list, min_dist=SQRT3 - 0.01):
    """Fast approximate overlap check using center distances."""
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.sqrt((inner_list[i][0] - inner_list[j][0]) ** 2 +
                              (inner_list[i][1] - inner_list[j][1]) ** 2)
            if dist < min_dist:
                return False
    return True


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


def sa_objective(params):
    """Direct objective: compute smooth required side length + overlap penalty."""
    inner_list = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        a = params[i * 3 + 2]
        inner_list.append([x, y, a])

    cx = params[N * 3]
    cy = params[N * 3 + 1]
    angle = params[N * 3 + 2]

    side = compute_required_side_smooth(inner_list, (cx, cy), angle, temp=0.05)

    penalty = 0.0
    min_dist = SQRT3 - 0.02
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ai = inner_list[i]
            xj, yj, aj = inner_list[j]
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2

    return side + 200.0 * penalty


def construct():
    start_time = time.time()

    # Multiple candidate topologies
    axial_coords_list = [
        # Standard hex cluster
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2),
        ],
        # Compact variant
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ],
        # Row-based
        [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (-1, 1), (3, 1)
        ],
        # Another row variant
        [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ],
        # Staggered
        [
            (0, 0), (1, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3), (2, 3)
        ],
    ]

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Try each topology with SA + polish
    for topo_idx, axial_coords in enumerate(axial_coords_list):
        if time.time() - start_time > DEADLINE - 20:
            break

        scale = 1.0 + 1e-6
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))

        initial_inner = [[x, y, 0.0] for x, y in centers]

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

        # SA parameters - fewer steps per topology to allow multiple topologies
        T0 = 0.5
        T_end = 0.002
        total_steps = 25000
        steps_per_check = 200

        np.random.seed(42 + topo_idx * 100)

        for step in range(total_steps):
            if time.time() - start_time > DEADLINE - 20:
                break

            T = T0 * (T_end / T0) ** (step / total_steps)

            new_params = current_params[:]
            move_type = np.random.randint(0, 5)

            if move_type < 3:
                idx = np.random.randint(0, N)
                base = idx * 3
                new_params[base] += np.random.uniform(-0.12, 0.12)
                new_params[base + 1] += np.random.uniform(-0.12, 0.12)
            elif move_type == 3:
                idx = np.random.randint(0, N)
                new_params[idx * 3 + 2] += np.random.uniform(-8, 8)
            else:
                if np.random.random() < 0.7:
                    new_params[N * 3] += np.random.uniform(-0.12, 0.12)
                    new_params[N * 3 + 1] += np.random.uniform(-0.12, 0.12)
                else:
                    new_params[N * 3 + 2] += np.random.uniform(-8, 8)

            new_obj = sa_objective(new_params)

            if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / max(T, 1e-10)):
                current_params = new_params
                current_obj = new_obj

                if step % steps_per_check == 0:
                    inner_l = []
                    for i in range(N):
                        inner_l.append([current_params[i*3], current_params[i*3+1], current_params[i*3+2]])
                    if fast_overlap_check(inner_l):
                        valid, inner_l2, center_l, angle_l = sa_validate(current_params)
                        if valid:
                            sa_side = compute_required_side(inner_l2, center_l, angle_l)
                            if sa_side < best_sa_obj:
                                best_sa_valid = True
                                best_sa_inner = inner_l2
                                best_sa_center = center_l
                                best_sa_angle = angle_l
                                best_sa_obj = sa_side

        # Update global best from this topology's SA
        if best_sa_valid and best_sa_obj < best_side:
            best_side = best_sa_obj
            best_inner = best_sa_inner
            best_outer_center = best_sa_center
            best_outer_angle = best_sa_angle

    # Final polish on the best solution found
    if time.time() - start_time < DEADLINE - 10 and best_inner is not None:
        try:
            flat = []
            for i, j, a in best_inner:
                flat.extend([i, j, a])
            flat.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

            def polish_obj(p):
                inner_l = []
                for i in range(N):
                    inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                cx = p[N*3]
                cy = p[N*3+1]
                ang = p[N*3+2]
                side = compute_required_side_smooth(inner_l, (cx, cy), ang, temp=0.01)
                pen = 0.0
                min_d = SQRT3 - 0.01
                for i in range(N):
                    for j in range(i+1, N):
                        d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                        if d < min_d:
                            pen += (min_d - d)**2
                return side + 500.0 * pen

            polish_res = minimize(
                polish_obj,
                flat,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 15000}
            )

            inner_l = []
            for i in range(N):
                inner_l.append([polish_res.x[i*3], polish_res.x[i*3+1], polish_res.x[i*3+2]])
            cx = polish_res.x[N*3]
            cy = polish_res.x[N*3+1]
            ang = polish_res.x[N*3+2]

            if check_all_disjoint(inner_l):
                side = compute_required_side(inner_l, (cx, cy), ang)
                if check_inside_outer(inner_l, (cx, cy), side, ang):
                    if side < best_side:
                        best_side = side
                        best_inner = inner_l
                        best_outer_center = (cx, cy)
                        best_outer_angle = ang
        except Exception:
            pass

    # Second polish pass with tighter tolerances
    if time.time() - start_time < DEADLINE - 5 and best_inner is not None:
        try:
            flat2 = []
            for i, j, a in best_inner:
                flat2.extend([i, j, a])
            flat2.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

            def polish2_obj(p):
                inner_l = []
                for i in range(N):
                    inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                cx = p[N*3]
                cy = p[N*3+1]
                ang = p[N*3+2]
                side = compute_required_side_smooth(inner_l, (cx, cy), ang, temp=0.005)
                pen = 0.0
                min_d = SQRT3 - 0.005
                for i in range(N):
                    for j in range(i+1, N):
                        d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                        if d < min_d:
                            pen += (min_d - d)**2
                return side + 800.0 * pen

            polish2_res = minimize(
                polish2_obj,
                flat2,
                method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 12000}
            )

            inner_l2 = []
            for i in range(N):
                inner_l2.append([polish2_res.x[i*3], polish2_res.x[i*3+1], polish2_res.x[i*3+2]])
            cx2 = polish2_res.x[N*3]
            cy2 = polish2_res.x[N*3+1]
            ang2 = polish2_res.x[N*3+2]

            if check_all_disjoint(inner_l2):
                side2 = compute_required_side(inner_l2, (cx2, cy2), ang2)
                if check_inside_outer(inner_l2, (cx2, cy2), side2, ang2):
                    if side2 < best_side:
                        best_side = side2
                        best_inner = inner_l2
                        best_outer_center = (cx2, cy2)
                        best_outer_angle = ang2
        except Exception:
            pass

    # Final verification and fallback
    if best_inner is None or not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        # Fallback to the first topology
        axial_coords = axial_coords_list[0]
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        best_inner = [[x, y, 0.0] for x, y in fallback_centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle
