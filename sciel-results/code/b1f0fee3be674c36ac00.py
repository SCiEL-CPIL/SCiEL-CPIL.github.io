"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        """Get 6 vertices of a unit regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def polygons_overlap_sat(v1, v2):
        """SAT overlap test for two convex polygons (each with 6 vertices)."""
        for poly in [v1, v2]:
            for i in range(len(poly)):
                edge = poly[(i + 1) % len(poly)] - poly[i]
                normal = np.array([-edge[1], edge[0]])
                if np.linalg.norm(normal) < 1e-12:
                    continue
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def check_all_disjoint(positions, angles):
        """Return True if all pairs are disjoint (non-overlapping)."""
        verts = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_overlap_sat(verts[i], verts[j]):
                    return False
        return True

    def compute_min_side(positions, angles, outer_angle_deg):
        """
        Minimum outer hexagon side length to contain all inner hexagons.
        Outer hexagon is centered at origin with given rotation.
        """
        all_verts = np.vstack([hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)])
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        # For a regular hexagon of side s, the apothem (inradius) is s*sqrt(3)/2
        # max_proj is the maximum projection onto any of the 6 normal directions
        # This equals the inradius of the containing hexagon
        return 2.0 * max_proj / math.sqrt(3)

    def get_config_score(positions, angles, outer_angle):
        """Return outer side length if valid (disjoint), else inf."""
        if not check_all_disjoint(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    def make_honeycomb(rows, cols):
        """
        Generate a honeycomb lattice of hexagons.
        For a regular hexagon of side 1, adjacent centers are sqrt(3) apart horizontally
        and 1.5 apart vertically (for pointy-top orientation).
        """
        positions = []
        angles = []
        for r in range(rows):
            for c in range(cols):
                x = c * math.sqrt(3)
                y = r * 1.5
                # Offset odd rows by sqrt(3)/2
                if r % 2 == 1:
                    x += math.sqrt(3) / 2
                positions.append((x, y))
                angles.append(0.0)
        # Center the configuration
        if positions:
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
            positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    def make_hex_ring(center_count, ring_count):
        """
        Place hexagons in a hexagonal ring pattern.
        center_count: number in center cluster
        ring_count: number in surrounding ring
        """
        positions = []
        angles = []
        # Place center hexagons in a small cluster
        if center_count >= 1:
            positions.append((0.0, 0.0))
            angles.append(0.0)
        if center_count >= 2:
            positions.append((math.sqrt(3), 0.0))
            angles.append(0.0)
        if center_count >= 3:
            positions.append((math.sqrt(3) / 2, 1.5))
            angles.append(0.0)
        if center_count >= 4:
            positions.append((-math.sqrt(3) / 2, 1.5))
            angles.append(0.0)
        if center_count >= 5:
            positions.append((-math.sqrt(3), 0.0))
            angles.append(0.0)
        if center_count >= 6:
            positions.append((-math.sqrt(3) / 2, -1.5))
            angles.append(0.0)
        if center_count >= 7:
            positions.append((math.sqrt(3) / 2, -1.5))
            angles.append(0.0)

        # Add ring hexagons around the cluster
        remaining = N - len(positions)
        if remaining > 0:
            # Generate ring positions at radius ~ 3*sqrt(3)/2 from center
            ring_radius = 3.0 * math.sqrt(3) / 2
            for i in range(remaining):
                a = 2 * math.pi * i / max(remaining, 1)
                x = ring_radius * math.cos(a)
                y = ring_radius * math.sin(a)
                positions.append((x, y))
                angles.append(0.0)

        # Center
        if positions:
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
            positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    def local_search(positions, angles, outer_angle, max_iters=5000):
        """
        Local search: try small perturbations of each hexagon's position
        to reduce the outer side length while maintaining disjointness.
        """
        pos = np.array(positions, dtype=float)
        ang = np.array(angles, dtype=float)
        best_score = get_config_score(list(map(tuple, pos)), list(ang), outer_angle)
        best_pos = pos.copy()

        if best_score == float('inf'):
            # First need to resolve overlaps
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                # Find an overlapping pair
                verts = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
                overlap_found = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if polygons_overlap_sat(verts[i], verts[j]):
                            # Push apart
                            d = pos[j] - pos[i]
                            dist = np.linalg.norm(d)
                            if dist < 1e-6:
                                d = np.array([0.1, 0.1])
                                dist = np.linalg.norm(d)
                            d_norm = d / dist
                            step = 0.05
                            new_pos = pos.copy()
                            new_pos[i] -= step * d_norm
                            new_pos[j] += step * d_norm
                            new_verts = [hex_vertices(new_pos[k][0], new_pos[k][1], ang[k]) for k in range(N)]
                            valid = True
                            for a in range(N):
                                for b in range(a + 1, N):
                                    if polygons_overlap_sat(new_verts[a], new_verts[b]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                pos = new_pos
                                overlap_found = True
                            break
                    if overlap_found:
                        break
                if not overlap_found:
                    break

            best_score = get_config_score(list(map(tuple, pos)), list(ang), outer_angle)
            best_pos = pos.copy()

        if best_score == float('inf'):
            return list(map(tuple, best_pos)), list(ang), outer_angle, best_score

        # Now optimize for minimal side length
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            improved = False
            for i in range(N):
                for dx in [-0.02, 0.0, 0.02]:
                    for dy in [-0.02, 0.0, 0.02]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = pos.copy()
                        new_pos[i] += np.array([dx, dy])
                        new_score = get_config_score(list(map(tuple, new_pos)), list(ang), outer_angle)
                        if new_score < best_score - 1e-8:
                            pos = new_pos
                            best_score = new_score
                            best_pos = pos.copy()
                            improved = True

            if not improved:
                break

        return list(map(tuple, best_pos)), list(ang), outer_angle, best_score

    def coordinate_descent(positions, angles, outer_angle, max_rounds=200):
        """
        Coordinate descent: optimize each hexagon's x and y separately.
        """
        pos = np.array(positions, dtype=float)
        ang = np.array(angles, dtype=float)
        best_score = get_config_score(list(map(tuple, pos)), list(ang), outer_angle)

        if best_score == float('inf'):
            return list(map(tuple, pos)), list(ang), outer_angle, best_score

        for rnd in range(max_rounds):
            if time.time() > deadline - 10:
                break

            improved = False
            for i in range(N):
                for axis in range(2):
                    # Try small steps in positive and negative direction
                    for step in [0.01, 0.005, 0.001]:
                        for direction in [1, -1]:
                            new_pos = pos.copy()
                            new_pos[i, axis] += direction * step
                            new_score = get_config_score(list(map(tuple, new_pos)), list(ang), outer_angle)
                            if new_score < best_score - 1e-8:
                                pos = new_pos
                                best_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                break

        return list(map(tuple, pos)), list(ang), outer_angle, best_score

    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    # Generate diverse initial configurations
    candidates = []

    # Honeycomb 4x3 (12 hexagons in 4 rows of 3)
    candidates.append(make_honeycomb(4, 3))
    # Honeycomb 3x4
    candidates.append(make_honeycomb(3, 4))
    # Honeycomb 2x6
    candidates.append(make_honeycomb(2, 6))
    # Honeycomb 6x2
    candidates.append(make_honeycomb(6, 2))
    # Hex ring: 7 center + 5 ring
    candidates.append(make_hex_ring(7, 5))
    # Hex ring: 4 center + 8 ring
    candidates.append(make_hex_ring(4, 8))
    # Hex ring: 1 center + 11 ring
    candidates.append(make_hex_ring(1, 11))

    # Also try with different outer angles for the best honeycomb
    base_pos, base_ang = make_honeycomb(4, 3)
    for oa in [0, 30, 60]:
        candidates.append((base_pos, base_ang))

    # Evaluate and optimize each candidate
    for idx, (pos, ang) in enumerate(candidates):
        if time.time() > deadline - 20:
            break

        # Try multiple outer angles
        for oa in [0, 30, 60]:
            if time.time() > deadline - 15:
                break

            score = get_config_score(pos, ang, oa)

            if score < best_score:
                best_score = score
                best_positions = pos
                best_angles = ang
                best_outer_angle = oa

            # If valid, try to improve
            if score < float('inf') and time.time() < deadline - 25:
                # Local search
                new_pos, new_ang, new_oa, new_score = local_search(pos, ang, oa, max_iters=3000)
                if new_score < best_score:
                    best_score = new_score
                    best_positions = new_pos
                    best_angles = new_ang
                    best_outer_angle = new_oa

                # Coordinate descent on the improved result
                if time.time() < deadline - 15:
                    cd_pos, cd_ang, cd_oa, cd_score = coordinate_descent(
                        new_pos, new_ang, new_oa, max_rounds=100)
                    if cd_score < best_score:
                        best_score = cd_score
                        best_positions = cd_pos
                        best_angles = cd_ang
                        best_outer_angle = cd_oa

    # Fallback if no valid configuration found
    if best_positions is None or best_score == float('inf'):
        positions = []
        angles = []
        for i in range(N):
            r = i // 4
            c = i % 4
            positions.append((c * 3.0, r * 3.0))
            angles.append(0.0)
        best_positions = positions
        best_angles = angles
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, angles, 0.0)

    # Final centering
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0.0, 0.0)] * N
        best_angles = [0.0] * N
        best_score = 10.0

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]

    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
