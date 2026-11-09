"""Hexagon packing: place disjoint unit regular hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time
import math

def construct():
    t0 = time.time()
    deadline = 145.0

    def hex_verts(cx, cy, r, angle_deg):
        a = np.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        pts = []
        for i in range(6):
            th = np.pi/3 * i + a
            x = cx + r * math.cos(th)
            y = cy + r * math.sin(th)
            pts.append([x, y])
        return np.array(pts)

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def point_in_convex_poly(p, poly):
        n = len(poly)
        for i in range(n):
            if cross(poly[i], poly[(i+1)%n], p) < -1e-9:
                return False
        return True

    def hex_overlap(v1, v2, eps=1e-10):
        # Check if two convex hexagons overlap (including touching)
        # Use separating axis theorem
        for verts in (v1, v2):
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 < min2 + eps or max2 < min1 + eps:
                    return False
        return True

    def is_valid_placement(inner_positions, outer_center, outer_side, outer_angle):
        # Check all inner hexagons are inside outer
        outer_verts = hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
        for (x, y, ang) in inner_positions:
            verts = hex_verts(x, y, 1.0, ang)
            for v in verts:
                if not point_in_convex_poly(v, outer_verts):
                    return False
        # Check pairwise disjoint
        n = len(inner_positions)
        for i in range(n):
            v1 = hex_verts(inner_positions[i][0], inner_positions[i][1], 1.0, inner_positions[i][2])
            for j in range(i+1, n):
                v2 = hex_verts(inner_positions[j][0], inner_positions[j][1], 1.0, inner_positions[j][2])
                if hex_overlap(v1, v2):
                    return False
        return True

    def try_packing(n_hex, max_iters=200):
        """Try to pack n_hex unit hexagons, return (positions, outer_side) or None."""
        best_side = None
        best_positions = None

        # Try different configurations
        # For n hexagons, the outer side must be at least something
        # Start with a grid-like arrangement and optimize

        # Strategy: use a triangular/hexagonal lattice arrangement
        # For n hexagons, find a compact arrangement

        # Hexagonal arrangement: center hex + rings
        # Ring k has 6k hexagons
        # Total: 1 + 6 + 12 + ... + 6k = 1 + 3k(k+1)

        # For arbitrary n, we can use a rectangular-ish hex grid
        # Hex grid with offset rows

        # Let's try a hexagonal close-packed arrangement
        # In a hex grid, adjacent hexagons (same orientation) are spaced sqrt(3) apart horizontally
        # and 1.5 * sqrt(3) = 3*sqrt(3)/2 vertically? No.
        # Regular hexagon with side 1: width = 2 (point to point), height = sqrt(3) (flat to flat)
        # If oriented with flat top/bottom: width = sqrt(3), height = 2
        # If oriented with point up: width = 2, height = sqrt(3)

        # Let's use flat-top hexagons (point up)
        # Horizontal spacing: 2
        # Vertical spacing: sqrt(3)
        # Offset rows: x offset by 1

        # For n hexagons, arrange in rows
        # Row with m hexagons: width = 2*(m-1) + 2 = 2m
        # Total width of arrangement: max over rows
        # Total height: sqrt(3) * (num_rows - 1) + sqrt(3) = sqrt(3) * num_rows

        # The enclosing regular hexagon must contain this arrangement
        # This is complex, so let's use a simpler approach:
        # Place hexagons in a hexagonal cluster and find minimal enclosing regular hexagon

        # For now, let's try a simple approach:
        # Place n hexagons in a hexagonal pattern centered at origin
        # Then find the minimal regular hexagon (with any rotation) that contains all of them

        # Hexagonal cluster positions (centered at origin, flat-top orientation)
        # Using axial coordinates
        def hex_cluster_positions(n):
            """Return positions of n hexagons in a compact hexagonal cluster."""
            if n == 1:
                return [(0, 0, 0)]

            positions = [(0.0, 0.0, 0.0)]
            remaining = n - 1

            ring = 1
            while remaining > 0:
                # Ring ring has 6*ring positions
                ring_size = min(6 * ring, remaining)
                # Place around the ring
                # For flat-top hexagons in axial coords:
                # Ring r: positions at distance r from center
                # In Cartesian (flat-top):
                #   x = 2 * (q + r/2)
                #   y = sqrt(3) * r
                # where (q, r) are axial coords with q + r = distance in some sense

                # Simpler: place 6*ring positions around a hexagon of "radius" ring
                # For flat-top hexagons:
                # The 6 corners of ring r are at angles 0, 60, 120, 180, 240, 300 degrees
                # at distance 2*r from center? No.

                # Let me use a different approach:
                # Generate hex grid positions using axial coordinates
                # q from -ring to ring, r from -ring to ring, with constraint |q| + |r| + |q+r| <= 2*ring... no

                # Actually for a hexagonal cluster of radius R (center + R rings):
                # Total hexagons = 1 + 3R(R+1)
                # Positions in axial coords (q, r) where -R <= q <= R, -R <= r <= R, -R <= q+r <= R
                # Cartesian: x = 2*q + r, y = sqrt(3)*r

                # Let me just generate all positions in order of distance from center
                all_pos = []
                for q in range(-ring, ring+1):
                    for r in range(-ring, ring+1):
                        if abs(q + r) <= ring:
                            x = 2.0 * q + r
                            y = math.sqrt(3) * r
                            all_pos.append((x, y))

                # Sort by distance from center
                all_pos.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])

                # Take the first ring_size positions (excluding center which is already placed)
                # Actually, the inner ring positions should be filled first
                # Let me just take all positions in this ring order

                # For simplicity, let me place them in a spiral-like order
                # Just take the first ring_size from the sorted list
                for (x, y) in all_pos[:ring_size]:
                    # Check if already placed
                    already = False
                    for (px, py, pa) in positions:
                        if abs(px - x) < 1e-9 and abs(py - y) < 1e-9:
                            already = True
                            break
                    if not already:
                        positions.append((x, y, 0.0))
                        remaining -= 1
                        if remaining == 0:
                            break

                ring += 1

            return positions

        # Find minimal enclosing regular hexagon for a set of points
        def min_enclosing_hexagon(positions, outer_center, max_rot_attempts=12):
            """Find minimal side length of regular hexagon containing all hexagons."""
            best_side = None
            for angle_deg in np.linspace(0, 30, max_rot_attempts):
                # For a regular hexagon with rotation angle, find minimal side
                # such that all inner hexagons fit inside

                # Binary search on side length
                lo = 1.0
                hi = 50.0

                # First check if hi works
                def check(side):
                    outer_verts = hex_verts(outer_center[0], outer_center[1], side, angle_deg)
                    for (x, y, ang) in positions:
                        verts = hex_verts(x, y, 1.0, ang)
                        for v in verts:
                            if not point_in_convex_poly(v, outer_verts):
                                return False
                    return True

                if not check(hi):
                    continue

                for _ in range(40):
                    mid = (lo + hi) / 2
                    if check(mid):
                        hi = mid
                    else:
                        lo = mid

                if best_side is None or hi < best_side:
                    best_side = hi

            return best_side

        # Try different arrangements
        # 1. Hexagonal cluster
        pos = hex_cluster_positions(n_hex)
        side = min_enclosing_hexagon(pos, (0.0, 0.0))
        if side is not None:
            # Verify
            inner = [[x, y, ang] for (x, y, ang) in pos]
            if is_valid_placement(inner, (0.0, 0.0), side, 0.0):
                best_side = side
                best_positions = inner

        # 2. Linear arrangement (for small n)
        if n_hex <= 6:
            # Place in a line
            pos = []
            for i in range(n_hex):
                pos.append((2.0 * i, 0.0, 0.0))
            # Center the arrangement
            cx = sum(p[0] for p in pos) / n_hex
            cy = 0.0
            pos = [(x - cx, y - cy, ang) for (x, y, ang) in pos]
            side = min_enclosing_hexagon(pos, (0.0, 0.0))
            if side is not None:
                inner = [[x, y, ang] for (x, y, ang) in pos]
                if is_valid_placement(inner, (0.0, 0.0), side, 0.0):
                    if best_side is None or side < best_side:
                        best_side = side
                        best_positions = inner

        # 3. Two-row arrangement
        if 2 <= n_hex <= 12:
            # Split into two rows
            half = (n_hex + 1) // 2
            other = n_hex - half
            pos = []
            for i in range(half):
                pos.append((2.0 * i - (half-1), 0.0, 0.0))
            for i in range(other):
                pos.append((2.0 * i - (other-1) + 1.0, math.sqrt(3), 0.0))
            cx = sum(p[0] for p in pos) / n_hex
            cy = sum(p[1] for p in pos) / n_hex
            pos = [(x - cx, y - cy, ang) for (x, y, ang) in pos]
            side = min_enclosing_hexagon(pos, (0.0, 0.0))
            if side is not None:
                inner = [[x, y, ang] for (x, y, ang) in pos]
                if is_valid_placement(inner, (0.0, 0.0), side, 0.0):
                    if best_side is None or side < best_side:
                        best_side = side
                        best_positions = inner

        if best_positions is not None and best_side is not None:
            return best_positions, best_side

        return None

    # Try increasing n until we can't fit anymore or time runs out
    best_result = None
    best_n = 0

    for n in range(1, 30):
        if time.time() > deadline:
            break

        result = try_packing(n)
        if result is not None:
            positions, side = result
            # Verify validity
            if is_valid_placement([[x, y, ang] for (x, y, ang) in positions], (0.0, 0.0), side, 0.0):
                if best_result is None or side < best_result[1]:
                    best_result = (positions, side)
                    best_n = n
        else:
            break

    if best_result is not None:
        positions, side = best_result
        inner = [[x, y, ang] for (x, y, ang) in positions]
        return inner, (0.0, 0.0), side, 0.0
    else:
        # Fallback: single hexagon
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 2.0, 0.0
