"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """
        Compute minimum outer side to contain all inner hexagons.
        """
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Pre-compute the 6 outward normals of a unit hexagon
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def hex_distance(p1, p2):
        """
        Exact minimum distance between two unit regular hexagons using SAT.
        Returns 0 if they overlap, positive if separated.
        """
        verts1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        verts2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        # Collect all edge normals (candidate separating axes)
        axes = []
        for i in range(6):
            for verts in (verts1, verts2):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                if length > 1e-12:
                    axes.append((-dy / length, dx / length))

        # Also add the direction between centers as a candidate axis
        dx_c = p2[0] - p1[0]
        dy_c = p2[1] - p1[1]
        dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
        if dist_c > 1e-12:
            axes.append((dx_c / dist_c, dy_c / dist_c))

        # For each axis, project both polygons and check for separation
        for nx, ny in axes:
            min1 = min(vx * nx + vy * ny for vx, vy in verts1)
            max1 = max(vx * nx + vy * ny for vx, vy in verts1)
            min2 = min(vx * nx + vy * ny for vx, vy in verts2)
            max2 = max(vx * nx + vy * ny for vx, vy in verts2)

            gap = max(min2 - max1, min1 - max2)
            if gap > 0:
                return gap

        # No separating axis found -> they overlap
        return 0.0

    def is_disjoint(p1, p2, tol=1e-9):
        """Check if two hexagons are disjoint (no overlap)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0 - tol:
            return True
        return hex_distance(p1, p2) > tol

    def all_disjoint(positions, tol=1e-9):
        """Check if all hexagons are pairwise disjoint."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if not is_disjoint(positions[i], positions[j], tol):
                    return False
        return True

    def simulated_annealing(initial_positions, outer_angle=0.0, max_iter=50000, start_temp=2.0, end_temp=0.01):
        """
        Simulated annealing to optimize positions of hexagons.
        Returns best positions found.
        """
        current_positions = [list(p) for p in initial_positions]
        current_side = compute_min_side(current_positions, (0.0, 0.0), outer_angle)
        best_positions = [list(p) for p in current_positions]
        best_side = current_side

        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iter)

        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break

            # Perturb a random hexagon
            idx = random.randint(0, N - 1)
            old_x, old_y, old_ang = current_positions[idx]

            # Small perturbation
            dx = random.gauss(0, 0.15)
            dy = random.gauss(0, 0.15)
            da = random.gauss(0, 5.0)

            new_x = old_x + dx
            new_y = old_y + dy
            new_ang = old_ang + da

            # Check if still disjoint from others
            candidate = [new_x, new_y, new_ang]
            disjoint = True
            for j in range(N):
                if j != idx and not is_disjoint(candidate, current_positions[j]):
                    disjoint = False
                    break

            if not disjoint:
                continue

            # Compute new side
            trial_positions = [list(p) for p in current_positions]
            trial_positions[idx] = candidate
            trial_side = compute_min_side(trial_positions, (0.0, 0.0), outer_angle)

            delta = trial_side - current_side
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_positions[idx] = candidate
                current_side = trial_side
                if current_side < best_side:
                    best_side = current_side
                    best_positions = [list(p) for p in current_positions]

            temp *= cooling

        return best_positions, best_side

    def build_rigid_cluster(ring_r, gap_angle, gap_dist):
        """
        Build a rigid cluster: 1 center + 6 ring + 4 outer.
        """
        positions = [[0.0, 0.0, 0.0]]
        for k in range(6):
            ang = math.radians(60 * k)
            x = ring_r * math.cos(ang)
            y = ring_r * math.sin(ang)
            positions.append([x, y, 0.0])
        for g in range(4):
            ang_deg = gap_angle + g * 90.0
            ang = math.radians(ang_deg)
            x = gap_dist * math.cos(ang)
            y = gap_dist * math.sin(ang)
            positions.append([x, y, 0.0])
        return positions

    # Main optimization
    best_inner = None
    best_side = float('inf')
    best_angle = 0.0

    # Try various initial configurations
    # 1. Rigid cluster approach
    ring_r_values = [sqrt3 + 0.05, sqrt3 + 0.1, sqrt3 + 0.15, sqrt3 + 0.2]
    gap_angle_values = [0, 15, 30, 45]
    gap_dist_values = [2.5, 2.8, 3.0, 3.2, 3.5]

    for ring_r in ring_r_values:
        if time.time() > deadline - 30.0:
            break

        for gap_angle in gap_angle_values:
            if time.time() > deadline - 30.0:
                break

            for gap_dist in gap_dist_values:
                if time.time() > deadline - 30.0:
                    break

                positions = build_rigid_cluster(ring_r, gap_angle, gap_dist)

                # Check if valid (all disjoint)
                if all_disjoint(positions):
                    side = compute_min_side(positions, (0.0, 0.0), 0.0)

                    if side < best_side:
                        best_side = side
                        best_inner = [list(p) for p in positions]
                        best_angle = 0.0

    # 2. Simulated annealing from multiple starting points
    start_configs = []

    # Add the best rigid cluster found
    if best_inner is not None:
        start_configs.append(best_inner)

    # Add some heuristic starting configurations
    # Configuration 1: Center + 6 around + 4 at corners
    config1 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = math.radians(60 * k)
        x = (sqrt3 + 0.1) * math.cos(ang)
        y = (sqrt3 + 0.1) * math.sin(ang)
        config1.append([x, y, 0.0])
    for g in range(4):
        ang_deg = 30 + g * 90
        ang = math.radians(ang_deg)
        x = 3.0 * math.cos(ang)
        y = 3.0 * math.sin(ang)
        config1.append([x, y, 0.0])
    if all_disjoint(config1):
        start_configs.append(config1)

    # Configuration 2: Different arrangement
    config2 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = math.radians(60 * k + 30)
        x = (sqrt3 + 0.15) * math.cos(ang)
        y = (sqrt3 + 0.15) * math.sin(ang)
        config2.append([x, y, 0.0])
    for g in range(4):
        ang_deg = 0 + g * 90
        ang = math.radians(ang_deg)
        x = 3.2 * math.cos(ang)
        y = 3.2 * math.sin(ang)
        config2.append([x, y, 0.0])
    if all_disjoint(config2):
        start_configs.append(config2)

    # Run simulated annealing from each starting configuration
    for start_pos in start_configs:
        if time.time() > deadline - 20.0:
            break

        # Try different outer angles
        for outer_angle in [0, 15, 30]:
            if time.time() > deadline - 20.0:
                break

            refined_positions, refined_side = simulated_annealing(start_pos, outer_angle, max_iter=20000, start_temp=1.5, end_temp=0.005)

            if refined_side < best_side:
                best_side = refined_side
                best_inner = refined_positions
                best_angle = outer_angle

    # 3. Final optimization of outer angle
    if best_inner is not None and time.time() < deadline - 5.0:
        for outer_angle in range(0, 60, 2):
            side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle

    # Ensure we have a valid result
    if best_inner is None:
        # Absolute fallback: safe configuration
        best_inner = []
        for i in range(N):
            if i == 0:
                best_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.5 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                best_inner.append([x, y, 0.0])
        best_side = compute_min_side(best_inner, (0.0, 0.0), 0.0)
        best_angle = 0.0

    # Final verification
    if not all_disjoint(best_inner):
        # Fallback to safe configuration
        safe_inner = []
        for i in range(N):
            if i == 0:
                safe_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.5 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                safe_inner.append([x, y, 0.0])
        best_inner = safe_inner
        best_side = compute_min_side(safe_inner, (0.0, 0.0), 0.0)
        best_angle = 0.0

    return best_inner, (0.0, 0.0), best_side, best_angle
