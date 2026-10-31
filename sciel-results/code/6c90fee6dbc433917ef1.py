import numpy as np
import math
import time
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # Safety margin before 150s kill


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
    inner: list of [x, y, angle_deg]
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-7):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    # Get axes from edges of v2
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

        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_cache = []
    for i in range(n):
        x, y, ang = inner[i]
        verts = get_hex_vertices(x, y, 1.0, math.radians(ang))
        verts_cache.append(verts)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_cache[i], verts_cache[j], eps):
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

    # 1. Define 11 hexagons on a hexagonal lattice (center + 10 neighbors)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # 2. Convert axial to Cartesian
    # Flat-top hexagons (angle 0 has vertex at right)
    # Axial to pixel for pointy-top (angle 30) is different.
    # Let's stick to standard "pointy top" or "flat top"?
    # Inspiration uses: x = 1.5 * q, y = sqrt(3) * (r + q/2). This is for "pointy top" hexagons?
    # Actually, for flat-top hexagons:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # This corresponds to hexagons where the top/bottom are flat?
    # Let's check vertices for angle 0: (1,0), (0.5, sqrt3/2)... This is pointy-top (vertex at right).
    # Wait, (1,0) is right. (0.5, 0.866) is top-right.
    # So angle 0 is "pointy right".
    
    epsilon = 1e-6
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]

    # Verify initial layout validity
    if not check_all_disjoint(initial_inner):
        # Fallback to sparse linear layout if lattice fails (shouldn't happen with scale > 1)
        initial_inner = [[i * 2.5, 0.0, 0.0] for i in range(N)]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Calculate initial centroid for outer center guess
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # --- Phase 1: Optimize Outer Hexagon (Center and Angle) ---
    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        side = compute_required_side(inner_list, (cx, cy), angle_deg)
        return side

    init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    best_outer_val = float('inf')
    best_outer_params = [cx_avg, cy_avg, 0.0]

    for init_angle in init_angles:
        if time.time() - start_time > DEADLINE - 60:
            break

        init_params = [cx_avg, cy_avg, float(init_angle)]
        try:
            res = minimize(
                lambda p: objective_outer(p, initial_inner),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500}
            )
            if res.fun < best_outer_val:
                best_outer_val = res.fun
                best_outer_params = res.x
        except Exception:
            pass

    if best_outer_val < float('inf'):
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        best_side = best_outer_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # --- Phase 2: Joint Optimization ---
    # We will optimize positions and angles of inner hexagons, plus outer center and angle.
    # To make this tractable, we'll do a few rounds of local search.
    # Variables: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle_outer]
    
    def objective_joint(params):
        if time.time() - start_time > DEADLINE - 10:
            return 1e10

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

        # Penalty for overlaps
        # Quick check: distance between centers must be > sqrt(3)
        # This is an approximation but fast.
        penalty = 0.0
        min_dist = SQRT3 - 0.01
        for i in range(N):
            xi, yi, _ = inner_list[i]
            for j in range(i + 1, N):
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000.0
        
        return side + penalty

    # Initialize params from best found so far
    params = []
    for i in range(N):
        params.extend(initial_inner[i])
    params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    # Run a few optimization steps
    num_rounds = 3
    for r in range(num_rounds):
        if time.time() - start_time > DEADLINE - 20:
            break

        try:
            res = minimize(
                objective_joint,
                params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 2000}
            )
            if res.fun < best_side:
                # Extract new best
                new_inner = []
                for i in range(N):
                    new_inner.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
                
                # Verify validity
                if check_all_disjoint(new_inner):
                    new_side = compute_required_side(new_inner, (res.x[N * 3], res.x[N * 3 + 1]), res.x[N * 3 + 2])
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = new_inner
                        best_outer_center = (res.x[N * 3], res.x[N * 3 + 1])
                        best_outer_angle = res.x[N * 3 + 2]
                        params = res.x
        except Exception:
            pass

    # Final verification and cleanup
    # Ensure we return a valid configuration
    if not check_all_disjoint(best_inner):
        # Fallback to initial if optimization broke it
        best_inner = [list(x) for x in initial_inner]
        best_side = compute_required_side(best_inner, (cx_avg, cy_avg), 0.0)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0

    # One last check: ensure inner hexagons are actually inside the computed outer side
    # If not, we might need to adjust, but compute_required_side should guarantee it by definition
    # unless numerical errors occur.
    
    return best_inner, best_outer_center, best_side, best_outer_angle
