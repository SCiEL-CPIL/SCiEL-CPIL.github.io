"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

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
        The outer hexagon has center (cx, cy), side length S, and rotation outer_angle.
        A point (px, py) is inside the outer hexagon iff for all 6 outward normals n_k,
        the projection of (px-cx, py-cy) rotated by -outer_angle onto n_k is <= S * sqrt(3)/2.
        So the required side is max over all points and all normals of proj / (sqrt(3)/2).
        """
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Pre-compute the 6 outward normals of a unit hexagon (inradius = sqrt(3)/2)
        # The normals point at angles 30, 90, 150, 210, 270, 330 degrees
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to outer hexagon's local frame
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def hex_distance(p1, p2):
        """
        Exact minimum distance between two unit regular hexagons.
        Returns 0 if they overlap, positive if separated.
        Uses Separating Axis Theorem on the convex polygons.
        """
        # Get vertices of both hexagons
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
                # Normal to this edge
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

            # If there's a gap, the distance is the gap
            gap = max(min2 - max1, min1 - max2)
            if gap > 0:
                return gap

        # No separating axis found -> they overlap
        return 0.0

    def check_overlap(p1, p2):
        """
        Return overlap penalty: 0 if disjoint, positive if overlapping.
        Uses exact SAT-based distance. If distance is 0, they overlap.
        We need a measure of how much they overlap. For the penalty,
        if they are disjoint (distance > 0), penalty is 0.
        If they overlap, we estimate the penetration depth.
        """
        # First quick check: if center distance > 2.0, definitely disjoint
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0:
            return 0.0

        # Use SAT to find minimum distance
        dist = hex_distance(p1, p2)
        if dist > 0:
            return 0.0
        else:
            # They overlap. Estimate penetration depth.
            # A rough estimate: if centers are close, penetration is significant.
            # Use (2.0 - d) as a proxy for penetration depth, clamped to be positive.
            # This is not exact but drives the optimizer to separate them.
            return max(0.0, 2.0 - d)

    def total_penalty(inner_positions):
        """Sum of overlap penalties for all pairs."""
        penalty = 0.0
        n = len(inner_positions)
        for i in range(n):
            for j in range(i + 1, n):
                penalty += check_overlap(inner_positions[i], inner_positions[j])
        return penalty

    def evaluate(positions, outer_angle):
        side = compute_min_side(positions, (0.0, 0.0), outer_angle)
        pen = total_penalty(positions)
        return side + 500.0 * pen

    # ---- Topological contact graph initialization ----
    # Start with a known dense configuration: central hex + 6 surrounding + 4 in gaps
    # This gives a good initial guess for the optimizer.

    def make_honeycomb_init():
        """Create 11 hexagons in a honeycomb-like pattern with safe spacing."""
        positions = []
        # Central hexagon at origin
        positions.append([0.0, 0.0, 0.0])

        # 6 hexagons around the center, touching it
        # Distance between centers of two touching unit hexagons (side-to-side) is sqrt(3)
        # But to be safe, use a slightly larger distance
        ring_dist = sqrt3 + 0.05
        for k in range(6):
            angle = math.radians(60 * k)
            x = ring_dist * math.cos(angle)
            y = ring_dist * math.sin(angle)
            positions.append([x, y, 0.0])

        # 4 additional hexagons in the gaps between the ring hexagons
        # These go at distance ~2*ring_dist from center, at 30, 150, 210, 330 degrees
        # Actually, the gaps between adjacent ring hexagons are at angles 30, 90, 150, 210, 270, 330
        # Pick 4 of them: 30, 150, 210, 330
        gap_angles = [30, 150, 210, 330]
        gap_dist = ring_dist * 2 * math.cos(math.pi / 6) + 0.05
        for angle_deg in gap_angles:
            angle = math.radians(angle_deg)
            x = gap_dist * math.cos(angle)
            y = gap_dist * math.sin(angle)
            positions.append([x, y, 0.0])

        return positions

    inner_positions = make_honeycomb_init()

    # Verify initial configuration has no overlaps
    init_penalty = total_penalty(inner_positions)
    if init_penalty > 1e-6:
        # Fallback to a safe spiral if honeycomb init overlaps
        inner_positions = []
        for i in range(N):
            if i == 0:
                inner_positions.append([0.0, 0.0, 0.0])
            else:
                r = 2.1 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                inner_positions.append([x, y, 0.0])

    # ---- Simulated Annealing with multi-start ----
    best_inner = [p[:] for p in inner_positions]
    best_score = float('inf')

    num_restarts = 8
    restarts_done = 0

    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        if restart == 0:
            current_positions = [p[:] for p in inner_positions]
        else:
            # Perturb the best solution found so far to create a new starting point
            current_positions = []
            for p in best_inner:
                new_p = p[:]
                new_p[0] += random.gauss(0, 0.5)
                new_p[1] += random.gauss(0, 0.5)
                new_p[2] += random.gauss(0, 15.0)
                current_positions.append(new_p)

        current_temp = 5.0
        cooling_rate = 0.995
        min_temp = 0.005
        current_score = evaluate(current_positions, 0.0)

        max_steps = 80000
        steps = 0
        no_improve_steps = 0
        best_in_run_score = current_score
        best_in_run = [p[:] for p in current_positions]

        while current_temp > min_temp and steps < max_steps and time.time() < deadline - 10.0:
            # Perturb one random hexagon
            idx = random.randint(0, N - 1)
            old_pos = current_positions[idx][:]
            new_pos = old_pos[:]

            # Move
            step_size = 0.15 * current_temp / 5.0 + 0.02
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            new_pos[0] += dx
            new_pos[1] += dy

            # Rotate
            d_angle = random.gauss(0, 5.0)
            new_pos[2] += d_angle

            # Apply change
            current_positions[idx] = new_pos

            new_score = evaluate(current_positions, 0.0)

            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / current_temp):
                current_score = new_score
                if current_score < best_in_run_score:
                    best_in_run_score = current_score
                    best_in_run = [p[:] for p in current_positions]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Revert
                current_positions[idx] = old_pos
                no_improve_steps += 1

            current_temp *= cooling_rate
            steps += 1

            # Early termination if stuck for too long
            if no_improve_steps > 20000:
                break

        # Update global best
        if best_in_run_score < best_score:
            best_score = best_in_run_score
            best_inner = [p[:] for p in best_in_run]

        restarts_done += 1

    # ---- Final optimization of outer angle ----
    best_side = float('inf')
    best_angle = 0.0

    # Try a range of outer angles
    angle_candidates = list(range(0, 60, 5)) + [3.0, 6.0, 9.0, 12.0, 18.0, 21.0, 24.0, 27.0, 33.0, 36.0, 39.0, 42.0, 48.0, 51.0, 54.0, 57.0]
    for outer_angle in angle_candidates:
        if time.time() > deadline - 5.0:
            break
        side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
        if side < best_side:
            best_side = side
            best_angle = outer_angle

    # Fine-tune the best angle with a local search
    if time.time() < deadline - 3.0:
        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            trial_angle = best_angle + random.gauss(0, 1.0)
            trial_side = compute_min_side(best_inner, (0.0, 0.0), trial_angle)
            if trial_side < best_side:
                best_side = trial_side
                best_angle = trial_angle

    # ---- Verify no overlap in the final configuration ----
    final_penalty = total_penalty(best_inner)
    if final_penalty > 1e-6:
        # Fallback to safe spiral
        safe_inner = []
        for i in range(N):
            if i == 0:
                safe_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.1 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                safe_inner.append([x, y, 0.0])

        best_side = float('inf')
        best_angle = 0.0
        for outer_angle in range(0, 60, 5):
            if time.time() > deadline - 2.0:
                break
            side = compute_min_side(safe_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle
        best_inner = safe_inner

    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
