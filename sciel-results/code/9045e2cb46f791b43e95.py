import numpy as np
import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3)

    # --- Geometry Helpers ---

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = np.zeros((6, 2))
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts[k, 0] = cx + side * math.cos(theta)
            verts[k, 1] = cy + side * math.sin(theta)
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point is inside hexagon."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a

        apothem = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(30 + k * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True

    def get_axes(verts):
        """Get edge normals for SAT."""
        axes = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            axes.append(normal)
        return axes

    def project(verts, axis):
        """Project vertices onto axis."""
        proj = np.dot(verts, axis)
        return proj.min(), proj.max()

    def sat_overlap(verts1, verts2):
        """
        Check if two convex polygons overlap using SAT.
        Returns (overlap, axis) where overlap is the minimum penetration depth
        and axis is the separating axis (normal).
        Returns (0, None) if no overlap.
        """
        min_overlap = float('inf')
        min_axis = None

        axes1 = get_axes(verts1)
        axes2 = get_axes(verts2)
        all_axes = axes1 + axes2

        for axis in all_axes:
            min1, max1 = project(verts1, axis)
            min2, max2 = project(verts2, axis)

            if max1 < min2 or max2 < min1:
                return 0.0, None  # Separated

            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                min_axis = axis

        # Ensure axis points from 1 to 2
        c1 = np.mean(verts1, axis=0)
        c2 = np.mean(verts2, axis=0)
        diff = c2 - c1
        if np.dot(diff, min_axis) < 0:
            min_axis = -min_axis

        return min_overlap, min_axis

    def min_outer_side(inner_list, outer_center, outer_angle):
        """Binary search for minimum outer side containing all inner hexagons."""
        cx, cy = outer_center
        lo = 1.0
        hi = 25.0

        def fits(side):
            for ix, iy, iang in inner_list:
                verts = hexagon_vertices(ix, iy, 1.0, iang)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, cx, cy, side, outer_angle):
                        return False
            return True

        if not fits(hi):
            return None
        if fits(lo):
            return lo

        for _ in range(30):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi + 1e-10

    def check_disjoint(inner_list, threshold=1e-6):
        """Check all pairs are disjoint using SAT."""
        n = len(inner_list)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, a1 = inner_list[i]
                x2, y2, a2 = inner_list[j]
                v1 = hexagon_vertices(x1, y1, 1.0, a1)
                v2 = hexagon_vertices(x2, y2, 1.0, a2)
                overlap, _ = sat_overlap(v1, v2)
                if overlap > threshold:
                    return False
        return True

    def evaluate(inner_list, outer_center, outer_angle):
        """Evaluate packing: return (side, valid)."""
        if not check_disjoint(inner_list):
            return None, False
        side = min_outer_side(inner_list, outer_center, outer_angle)
        if side is None:
            return None, False
        return side, True

    # --- Simulated Annealing Optimizer ---

    def sa_optimize(init_inner, outer_angle, max_iters=5000, t0=1.0, t_end=0.001):
        """
        Simulated annealing to minimize outer hexagon side length.
        init_inner: list of [x, y, angle_deg]
        Returns: (best_inner, best_side, valid)
        """
        n = len(init_inner)
        outer_center = (0.0, 0.0)

        # Initial state
        current = [list(p) for p in init_inner]
        # Center the configuration
        cx_mean = sum(p[0] for p in current) / n
        cy_mean = sum(p[1] for p in current) / n
        current = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in current]

        # Evaluate initial
        cur_side, cur_valid = evaluate(current, outer_center, outer_angle)
        if cur_side is None:
            cur_side = 25.0  # Penalty for invalid

        best_side = cur_side
        best_config = [list(p) for p in current]

        # Temperature schedule
        T = t0
        cooling = (t_end / t0) ** (1.0 / max_iters)

        rng = np.random.default_rng(42)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Generate perturbation
            candidate = [list(p) for p in current]
            # Pick a random hexagon to move
            idx = rng.integers(0, n)
            dx = rng.normal(0, 0.5)
            dy = rng.normal(0, 0.5)
            da = rng.normal(0, 10.0)  # degrees
            candidate[idx][0] += dx
            candidate[idx][1] += dy
            candidate[idx][2] += da

            # Re-center
            cx_mean = sum(p[0] for p in candidate) / n
            cy_mean = sum(p[1] for p in candidate) / n
            candidate = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in candidate]

            # Evaluate
            new_side, new_valid = evaluate(candidate, outer_center, outer_angle)
            if new_side is None:
                new_side = 25.0

            # Acceptance criterion
            delta = new_side - cur_side
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current = candidate
                cur_side = new_side
                if cur_side < best_side:
                    best_side = cur_side
                    best_config = [list(p) for p in current]

            T *= cooling

        return best_config, best_side, (best_side < 25.0)

    # --- Initial Configuration Generation ---

    def generate_hex_ring(n):
        positions = [(0, 0)]
        ring = 1
        while len(positions) < n:
            q, r = ring, 0
            for side_idx in range(6):
                for step in range(ring):
                    positions.append((q, r))
                    dq = [1, 0, -1, -1, 0, 1][side_idx]
                    dr = [0, -1, -1, 0, 1, 1][side_idx]
                    q += dq
                    r += dr
            ring += 1
        return positions[:n]

    def axial_to_cartesian(q, r, spacing):
        x = spacing * (2.0 * q + r)
        y = spacing * sqrt3 * r
        return x, y

    best_side = float('inf')
    best_result = None

    # --- Strategy 1: Simulated Annealing from Hexagonal Lattice ---
    axial_pos = generate_hex_ring(N)

    for spacing in [1.8, 1.7, 1.6]:
        if time.time() > deadline - 30:
            break
        for outer_angle in [0.0, 30.0]:
            if time.time() > deadline - 25:
                break

            init_inner = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, spacing)
                init_inner.append([x, y, 0.0])

            # Center cluster
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]

            # Run SA
            best_config, best_s, valid = sa_optimize(centered_init, outer_angle, max_iters=3000)
            if valid and best_s < best_side:
                best_side = best_s
                best_result = (best_config, (0.0, 0.0), best_s, outer_angle)

    # --- Strategy 2: Random Starts with SA ---
    if time.time() < deadline - 40:
        rng = np.random.default_rng(123)
        for trial in range(5):
            if time.time() > deadline - 20:
                break

            init_inner = []
            for _ in range(N):
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                a = rng.uniform(0, 360)
                init_inner.append([x, y, a])

            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]

            best_config, best_s, valid = sa_optimize(centered_init, 0.0, max_iters=2000)
            if valid and best_s < best_side:
                best_side = best_s
                best_result = (best_config, (0.0, 0.0), best_s, 0.0)

    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_result
