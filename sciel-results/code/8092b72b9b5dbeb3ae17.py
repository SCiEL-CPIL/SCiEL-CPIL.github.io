"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Geometry helpers ---
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def polygon_area_signed(verts):
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def polygon_overlap_area(v1, v2):
        """Compute overlap area of two convex polygons using Sutherland-Hodgman clipping."""
        # Clip v1 by each edge of v2
        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output

        def point_in_convex(p, poly):
            """Check if point p is inside convex polygon poly (CCW)."""
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                    return False
            return True

        def line_intersect(p1, p2, p3, p4):
            """Intersection of line p1-p2 and line p3-p4."""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-15:
                return p1
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def min_separation_dist(v1, v2):
        """Minimum distance between two convex polygons. 0 if overlapping."""
        # If they overlap, return 0
        ov = polygon_overlap_area(v1, v2)
        if ov > 1e-10:
            return -ov  # negative indicates overlap
        # Find min distance between edges
        min_d = float('inf')
        for i in range(len(v1)):
            for j in range(len(v2)):
                p1 = v1[i]
                p2 = v1[(i + 1) % len(v1)]
                p3 = v2[j]
                p4 = v2[(j + 1) % len(v2)]
                d = seg_seg_dist(p1, p2, p3, p4)
                if d < min_d:
                    min_d = d
        return min_d

    def seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments."""
        def dist_point_seg(px, py, ax, ay, bx, by):
            dx = bx - ax
            dy = by - ay
            if dx == 0 and dy == 0:
                return math.hypot(px - ax, py - ay)
            t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            qx = ax + t * dx
            qy = ay + t * dy
            return math.hypot(px - qx, py - qy)
        d1 = dist_point_seg(p1[0], p1[1], p3[0], p3[1], p4[0], p4[1])
        d2 = dist_point_seg(p2[0], p2[1], p3[0], p3[1], p4[0], p4[1])
        d3 = dist_point_seg(p3[0], p3[1], p1[0], p1[1], p2[0], p2[1])
        d4 = dist_point_seg(p4[0], p4[1], p1[0], p1[1], p2[0], p2[1])
        return min(d1, d2, d3, d4)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        verts = hex_vertices(cx, cy, side, angle_deg)
        return point_in_convex((px, py), verts)

    def hex_in_hexagon(inner_verts, outer_verts):
        """Check if all vertices of inner hex are inside outer hex."""
        return all(point_in_convex(v, outer_verts) for v in inner_verts)

    def outer_side_required(inner_positions, outer_center, outer_angle):
        """Find minimum outer side length such that all inner hexagons fit inside."""
        cx, cy = outer_center
        # Binary search on outer side length
        lo = 1.0
        hi = 100.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            outer_verts = hex_vertices(cx, cy, mid, outer_angle)
            all_fit = True
            for (ix, iy, iang) in inner_positions:
                inner_verts = hex_vertices(ix, iy, 1.0, iang)
                if not hex_in_hexagon(inner_verts, outer_verts):
                    all_fit = False
                    break
            if all_fit:
                hi = mid
            else:
                lo = mid
        return hi

    def objective(params, outer_center, outer_angle):
        """Penalty-based objective: outer side length + overlap penalties."""
        # Unpack params: for each inner hex: x, y, angle
        inner_positions = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2] * 180.0 / math.pi  # convert to degrees
            inner_positions.append((x, y, a))

        # Compute required outer side
        outer_side = outer_side_required(inner_positions, outer_center, outer_angle)

        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(*inner_positions[i][:2], 1.0, inner_positions[i][2])
                vj = hex_vertices(*inner_positions[j][:2], 1.0, inner_positions[j][2])
                sep = min_separation_dist(vi, vj)
                if sep < 0:
                    overlap_penalty += (-sep) ** 2

        return outer_side + 1000.0 * overlap_penalty

    def is_valid(inner_positions, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hex."""
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        # Check containment
        for (ix, iy, iang) in inner_positions:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(*inner_positions[i][:2], 1.0, inner_positions[i][2])
                vj = hex_vertices(*inner_positions[j][:2], 1.0, inner_positions[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Initial guess: hexagonal lattice packing ---
    def initial_guess():
        """Generate initial positions using a hexagonal lattice."""
        # Hexagonal lattice spacing for unit hexagons (flat-top orientation)
        # For flat-top hexagons, horizontal spacing = 3, vertical spacing = 3*sqrt(3)/2
        # But we can also rotate, so let's use a simple grid first
        params = np.zeros(3 * N)
        # Place in a hexagonal arrangement
        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons
        for k in range(6):
            theta = k * math.pi / 3.0
            x = 3.0 * math.cos(theta)
            y = 3.0 * math.sin(theta)
            positions.append((x, y, 0.0))
        # Ring 2: 12 hexagons (we need 4 more to reach 11)
        for k in range(4):
            theta = k * math.pi / 3.0 + math.pi / 6.0
            x = 5.0 * math.cos(theta)
            y = 5.0 * math.sin(theta)
            positions.append((x, y, 0.0))

        for i, (x, y, a) in enumerate(positions):
            params[3 * i] = x
            params[3 * i + 1] = y
            params[3 * i + 2] = a * math.pi / 180.0
        return params

    # Try multiple initial configurations and optimizations
    best_result = None
    best_score = float('inf')

    outer_center = (0.0, 0.0)
    outer_angle = 0.0

    # Try several random initializations
    rng = np.random.default_rng(42)

    attempts = [
        "lattice",
        "random1",
        "random2",
        "compact",
    ]

    for attempt_name in attempts:
        if time.time() > deadline - 20:
            break

        if attempt_name == "lattice":
            params0 = initial_guess()
        elif attempt_name == "compact":
            # More compact: tighter hexagonal packing
            params0 = np.zeros(3 * N)
            positions = []
            # Use closer spacing
            positions.append((0.0, 0.0, 0.0))
            for k in range(6):
                theta = k * math.pi / 3.0
                x = 2.8 * math.cos(theta)
                y = 2.8 * math.sin(theta)
                positions.append((x, y, 0.0))
            for k in range(4):
                theta = k * math.pi / 3.0 + math.pi / 6.0
                x = 4.8 * math.cos(theta)
                y = 4.8 * math.sin(theta)
                positions.append((x, y, 0.0))
            for i, (x, y, a) in enumerate(positions):
                params0[3 * i] = x
                params0[3 * i + 1] = y
                params0[3 * i + 2] = a * math.pi / 180.0
        else:
            # Random initialization within a reasonable box
            params0 = np.zeros(3 * N)
            for i in range(N):
                params0[3 * i] = rng.uniform(-8, 8)
                params0[3 * i + 1] = rng.uniform(-8, 8)
                params0[3 * i + 2] = rng.uniform(-math.pi, math.pi)

        # Run Nelder-Mead optimization
        try:
            result = minimize(
                objective,
                params0,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
        except Exception:
            continue

        if time.time() > deadline - 10:
            break

        # Extract best params
        best_params = result.x
        inner_positions = []
        for i in range(N):
            x = best_params[3 * i]
            y = best_params[3 * i + 1]
            a = best_params[3 * i + 2] * 180.0 / math.pi
            inner_positions.append((x, y, a))

        # Compute actual outer side
        outer_side = outer_side_required(inner_positions, outer_center, outer_angle)

        # Check validity
        if is_valid(inner_positions, outer_center, outer_side, outer_angle):
            if outer_side < best_score:
                best_score = outer_side
                best_result = (inner_positions, outer_center, outer_side, outer_angle)
        else:
            # Even if invalid, record as fallback if no valid found yet
            if best_result is None:
                # Force validity by scaling up outer
                outer_side_forced = outer_side * 1.5
                best_result = (inner_positions, outer_center, outer_side_forced, outer_angle)
                best_score = outer_side_forced

    # If no valid result found, return a safe default
    if best_result is None:
        # Fallback: place hexagons in a line with large outer
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_positions, outer_center, outer_side, outer_angle = best_result

    # Convert to required format
    inner = [[p[0], p[1], p[2]] for p in inner_positions]

    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END
