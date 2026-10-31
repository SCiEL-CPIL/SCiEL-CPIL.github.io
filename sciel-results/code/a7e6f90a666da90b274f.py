import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon (flat-top, vertex at angle)."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def hex_inradius(side):
    return side * SQRT3 / 2.0

def polygons_disjoint(v1, v2, eps=1e-9):
    """
    Check if two convex polygons are disjoint (no overlap, touching allowed).
    Uses SAT: if there exists a separating axis, they are disjoint.
    v1, v2: lists of (x, y)
    """
    for verts in (v1, v2):
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            # edge vector
            ex, ey = x2 - x1, y2 - y1
            # normal
            nx, ny = -ey, ex
            # project both polygons onto this normal
            min1 = max1 = None
            for px, py in v1:
                p = px * nx + py * ny
                if min1 is None or p < min1:
                    min1 = p
                if max1 is None or p > max1:
                    max1 = p
            min2 = max2 = None
            for px, py in v2:
                p = px * nx + py * ny
                if min2 is None or p < min2:
                    min2 = p
                if max2 is None or p > max2:
                    max2 = p
            # if separated (with tolerance)
            if max1 + eps < min2 or max2 + eps < min1:
                return True
    return False

def overlap_amount(v1, v2):
    """
    Approximate overlap: 0 if disjoint, else a positive penalty.
    We use the maximum penetration depth approximation via SAT:
    find the axis with the smallest separation; if negative, that's penetration.
    """
    min_pen = 0.0  # we track the least negative (closest to separating)
    found_separating = False
    for verts in (v1, v2):
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            ex, ey = x2 - x1, y2 - y1
            nx, ny = -ey, ex
            # normalize normal
            nlen = math.hypot(nx, ny)
            if nlen < 1e-12:
                continue
            nx /= nlen
            ny /= nlen
            min1 = max1 = None
            for px, py in v1:
                p = px * nx + py * ny
                if min1 is None or p < min1:
                    min1 = p
                if max1 is None or p > max1:
                    max1 = p
            min2 = max2 = None
            for px, py in v2:
                p = px * nx + py * ny
                if min2 is None or p < min2:
                    min2 = p
                if max2 is None or p > max2:
                    max2 = p
            # separation = min2 - max1 (positive means v2 is beyond v1 on this axis)
            sep1 = min2 - max1
            sep2 = min1 - max2
            sep = min(sep1, sep2)  # the actual gap (negative if overlapping)
            if sep > 1e-9:
                found_separating = True
                break
            # sep <= 0 means overlap on this axis; penetration = -sep
            pen = -sep
            if pen > min_pen:
                min_pen = pen
        if found_separating:
            break
    if found_separating:
        return 0.0
    return min_pen

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
    # Outward normals of a standard (angle 0) hexagon: at 30, 90, 150, 210, 270, 330
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
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    # Initial layout: hexagonal lattice, 3-4-3 arrangement
    # Axial coordinates for a compact cluster
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 4 hexagons (partial)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # Convert axial to Cartesian (flat-top)
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))

    # Initial: all hexagons aligned, angle 0
    # State vector: [x0, y0, a0, x1, y1, a1, ..., xN-1, yN-1, aN-1, cx, cy, cangle]
    # Total: N*3 + 3
    state0 = []
    for i in range(N):
        state0.append(centers[i][0])
        state0.append(centers[i][1])
        state0.append(0.0)
    # Outer center: centroid
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    state0.append(cx_avg)
    state0.append(cy_avg)
    state0.append(0.0)

    state0 = np.array(state0, dtype=float)

    def objective(state):
        """
        Objective: outer side length + penalty for overlaps.
        state: [x0, y0, a0, ..., xN-1, yN-1, aN-1, cx, cy, cangle]
        """
        inner = []
        for i in range(N):
            x = state[i*3]
            y = state[i*3 + 1]
            a = state[i*3 + 2] * 180.0 / math.pi  # radians to degrees
            inner.append((x, y, a))

        oc = (state[N*3], state[N*3 + 1])
        oa = state[N*3 + 2] * 180.0 / math.pi

        side = compute_required_side(inner, oc, oa)

        # Overlap penalty: check all pairs
        penalty = 0.0
        verts_list = [get_hex_vertices(x, y, 1.0, a * math.pi / 180.0) for (x, y, a) in inner]
        for i in range(N):
            for j in range(i + 1, N):
                pen = overlap_amount(verts_list[i], verts_list[j])
                if pen > 0:
                    penalty += pen ** 2

        # Strong penalty for overlaps
        return side + 10.0 * penalty

    # Try Nelder-Mead optimization
    best_state = state0.copy()
    best_obj = objective(state0)

    # Run Nelder-Mead
    try:
        res = minimize(
            objective,
            state0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-5,
                'fatol': 1e-7,
                'adaptive': True
            }
        )
        if res.fun < best_obj:
            best_state = res.x
            best_obj = res.fun
    except Exception:
        pass

    if time.time() > deadline - 10:
        # Time to finalize
        pass
    else:
        # Try a few more restarts with slight perturbations if time permits
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            perturbed = best_state + rng.normal(0, 0.1, size=best_state.shape)
            try:
                res2 = minimize(
                    objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-5,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
                if res2.fun < best_obj:
                    best_state = res2.x
                    best_obj = res2.fun
            except Exception:
                pass

    # Extract best solution
    inner = []
    for i in range(N):
        x = best_state[i*3]
        y = best_state[i*3 + 1]
        a = best_state[i*3 + 2] * 180.0 / math.pi
        inner.append([x, y, a])

    oc = (float(best_state[N*3]), float(best_state[N*3 + 1]))
    oa = float(best_state[N*3 + 2]) * 180.0 / math.pi

    # Compute the actual side length (without penalty)
    side = compute_required_side(inner, oc, oa)

    # Verify no overlaps; if there are, we need to handle
    verts_list = [get_hex_vertices(x, y, 1.0, a * math.pi / 180.0) for (x, y, a) in inner]
    has_overlap = False
    for i in range(N):
        for j in range(i + 1, N):
            if not polygons_disjoint(verts_list[i], verts_list[j], eps=1e-7):
                has_overlap = True
                break
        if has_overlap:
            break

    if has_overlap:
        # Fallback to the initial lattice layout which is guaranteed disjoint
        # (with a tiny scale factor to ensure strict disjointness)
        epsilon = 1e-6
        scale = 1.0 + epsilon
        centers2 = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers2.append((x, y))
        inner = [[x, y, 0.0] for x, y in centers2]
        cx_avg2 = sum(c[0] for c in centers2) / N
        cy_avg2 = sum(c[1] for c in centers2) / N
        oc = (cx_avg2, cy_avg2)
        oa = 0.0
        side = compute_required_side(inner, oc, oa)

    # Add tiny epsilon to side to ensure validity
    side += 1e-9

    return inner, oc, side, oa
