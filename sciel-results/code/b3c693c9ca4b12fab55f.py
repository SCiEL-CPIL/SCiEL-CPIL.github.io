import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Returns 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts

    def point_in_hexagon(px, py, hx, hy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (hx, hy)."""
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        inradius = side * sqrt3 / 2
        for k in range(6):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            if lx * nx + ly * ny > inradius + 1e-9:
                return False
        return True

    def hex_radius_from_center(px, py, hx, hy, side, angle_deg):
        """Compute the required side length of hexagon centered at (hx,hy) with angle to contain point (px,py)."""
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = lx * nx + ly * ny
            if proj > max_proj:
                max_proj = proj
        # inradius = side * sqrt3/2, so side = max_proj / (sqrt3/2) = 2*max_proj/sqrt3
        return 2.0 * max_proj / sqrt3

    def point_in_polygon(point, polygon):
        """Ray casting algorithm to determine if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if (p1x <= xinters or p2x <= xinters) and (p1x != p2x):
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def orientation(p, q, r):
        """Find orientation of triangle formed by p, q, r."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2

    def do_intersect(p1, q1, p2, q2):
        """Check if segment p1-q1 intersects with segment p2-q2."""
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    def edges_intersect(v1, v2):
        """Check if any edge of v1 intersects any edge of v2."""
        for i in range(6):
            p1 = v1[i]
            q1 = v1[(i + 1) % 6]
            for j in range(6):
                p2 = v2[j]
                q2 = v2[(j + 1) % 6]
                if do_intersect(p1, q1, p2, q2):
                    return True
        return False

    def hexes_overlap(v1, v2):
        """Check if two sets of vertices (hexagons) overlap."""
        for p in v1:
            if point_in_polygon(p, v2):
                return True
        for p in v2:
            if point_in_polygon(p, v1):
                return True
        if edges_intersect(v1, v2):
            return True
        return False

    def check_validity(params):
        """
        Check if the configuration is valid (all inner hexagons inside outer, no overlaps).
        Returns (is_valid, max_violation).
        """
        outer_cx, outer_cy, outer_side, outer_angle = params[0], params[1], params[2], params[3]
        
        if outer_side < 0.5:
            return False, 1.0

        inner_verts_list = []
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = get_hex_vertices(ix, iy, 1.0, iang)
            inner_verts_list.append(verts)
            
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    return False, 0.1

        for i in range(N):
            for j in range(i + 1, N):
                cx1 = params[4 + i * 3]
                cy1 = params[5 + i * 3]
                cx2 = params[4 + j * 3]
                cy2 = params[5 + j * 3]
                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
                if dist_sq > 4.5:
                    continue
                if hexes_overlap(inner_verts_list[i], inner_verts_list[j]):
                    return False, 0.1

        return True, 0.0

    def objective(params):
        """Objective: outer side length + penalties for violations."""
        if time.time() > deadline:
            return 1e10

        outer_cx, outer_cy, outer_side, outer_angle = params[0], params[1], params[2], params[3]
        
        if outer_side < 0.5:
            return 1e10

        # Check containment
        penalty = 0.0
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = get_hex_vertices(ix, iy, 1.0, iang)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    # Estimate how far outside
                    req_side = hex_radius_from_center(vx, vy, outer_cx, outer_cy, 1.0, outer_angle)
                    if req_side > outer_side:
                        penalty += (req_side - outer_side) * 10.0
                    else:
                        penalty += 1.0

        if penalty > 50.0:
            return outer_side + penalty

        # Check overlaps
        inner_verts_list = []
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = get_hex_vertices(ix, iy, 1.0, iang)
            inner_verts_list.append(verts)

        overlap_count = 0
        for i in range(N):
            for j in range(i + 1, N):
                cx1 = params[4 + i * 3]
                cy1 = params[5 + i * 3]
                cx2 = params[4 + j * 3]
                cy2 = params[5 + j * 3]
                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
                if dist_sq > 4.5:
                    continue
                if hexes_overlap(inner_verts_list[i], inner_verts_list[j]):
                    overlap_count += 1
                    if overlap_count > 5:
                        return outer_side + 100.0

        if overlap_count > 0:
            return outer_side + 10.0 * overlap_count

        return outer_side

    # Generate multiple initial configurations
    def make_honeycomb_init(scale=1.0, offset_x=0.0, offset_y=0.0, angle=0.0, jitter=0.0):
        """Create a honeycomb lattice initial configuration."""
        params = [0.0, 0.0, 6.0, 0.0]
        for row in range(3):
            for col in range(4):
                x = col * sqrt3 * scale + (sqrt3 * scale / 2 if row % 2 == 1 else 0)
                y = row * 1.5 * scale
                x -= 2.6 * scale + offset_x
                y -= 1.5 * scale + offset_y
                if jitter > 0:
                    x += np.random.normal(0, jitter)
                    y += np.random.normal(0, jitter)
                params.extend([x, y, angle])
        return params

    def make_ring_init(radius, angle_offset=0.0):
        """Create a ring configuration: 1 center + 6 inner ring + 5 outer."""
        params = [0.0, 0.0, 6.0, 0.0]
        # Center
        params.extend([0.0, 0.0, 0.0])
        # Inner ring of 6
        for k in range(6):
            a = angle_offset + k * math.pi / 3
            x = radius * math.cos(a)
            y = radius * math.sin(a)
            params.extend([x, y, 0.0])
        # Outer 5
        for k in range(5):
            a = angle_offset + math.pi/6 + k * 2*math.pi/5
            r2 = radius * 2.0
            x = r2 * math.cos(a)
            y = r2 * math.sin(a)
            params.extend([x, y, 0.0])
        return params

    # Try multiple initial configurations with local optimization
    best_params = None
    best_side = float('inf')

    initial_configs = []
    # Honeycomb variants
    for scale in [0.9, 1.0, 1.1]:
        for angle in [0.0, 30.0]:
            initial_configs.append(make_honeycomb_init(scale=scale, angle=angle, jitter=0.05))
    # Ring variants
    for radius in [1.8, 2.0, 2.2]:
        for aoff in [0.0, math.pi/6]:
            initial_configs.append(make_ring_init(radius, aoff))

    # Also try a compact cluster
    def make_cluster_init():
        params = [0.0, 0.0, 6.0, 0.0]
        positions = [
            (0, 0),
            (sqrt3, 0), (-sqrt3, 0),
            (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
            (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
            (2*sqrt3, 0),
            (sqrt3, 3.0), (-sqrt3, 3.0),
            (sqrt3, -3.0), (-sqrt3, -3.0),
            (0, 0),  # duplicate, will be adjusted
        ]
        # Use a better arrangement: 3x4 grid shifted
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * sqrt3 + (sqrt3/2 if row % 2 == 1 else 0)
                y = row * 1.5
                x -= 2.6
                y -= 1.5
                positions.append((x, y))
        for x, y in positions:
            params.extend([x, y, 0.0])
        return params

    initial_configs.append(make_cluster_init())

    bounds = [
        (-6.0, 6.0),   # outer_cx
        (-6.0, 6.0),   # outer_cy
        (2.5, 12.0),   # outer_side
        (0.0, 60.0)    # outer_angle
    ]
    for i in range(N):
        bounds.append((-6.0, 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((0.0, 60.0))

    # Try each initial configuration with Nelder-Mead
    for init_idx, init_params in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break

        try:
            result = minimize(
                objective, init_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass

        # Also try BFGS with finite differences for some configs
        if init_idx < 4 and time.time() > deadline - 20:
            break
        if init_idx < 4:
            try:
                result2 = minimize(
                    objective, init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-5, 'gtol': 1e-4}
                )
                if result2.fun < best_side:
                    best_side = result2.fun
                    best_params = result2.x
            except Exception:
                pass

    # If we have a good result, do a final refinement
    if best_params is not None and best_side < float('inf') and time.time() < deadline - 15:
        try:
            result_final = minimize(
                objective, best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if result_final.fun < best_side:
                best_side = result_final.fun
                best_params = result_final.x
        except Exception:
            pass

    # Verify and return
    if best_params is None or best_side >= float('inf'):
        # Fallback: return a known valid configuration
        inner = []
        for row in range(3):
            for col in range(4):
                x = col * sqrt3 + (sqrt3 / 2 if row % 2 == 1 else 0)
                y = row * 1.5
                x -= 2.6
                y -= 1.5
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 8.0, 0.0

    # Check validity of best result
    is_valid, _ = check_validity(best_params)
    if not is_valid:
        # Try to fix by slightly expanding
        outer_cx, outer_cy, outer_side, outer_angle = best_params[0], best_params[1], best_params[2], best_params[3]
        # Expand outer side until valid
        for trial in range(20):
            new_side = outer_side + 0.1 * (trial + 1)
            test_params = list(best_params)
            test_params[2] = new_side
            is_valid, _ = check_validity(test_params)
            if is_valid:
                best_params = test_params
                break

    outer_cx, outer_cy, outer_side, outer_angle = best_params[0], best_params[1], best_params[2], best_params[3]

    inner = []
    for i in range(N):
        ix = best_params[4 + i * 3]
        iy = best_params[5 + i * 3]
        iang = best_params[6 + i * 3]
        inner.append([ix, iy, iang])

    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
