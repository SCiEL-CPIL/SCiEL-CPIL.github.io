"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Helper: convert axial coordinates to Cartesian
    # For flat-top hexagons: x = 1.5*q, y = sqrt(3)*(r + q/2)
    # For pointy-top: x = sqrt(3)*(q + r/2), y = 1.5*r
    # Let's use a standard hex grid where adjacent centers are distance 2 apart.
    # Using: x = 2*q + r, y = sqrt(3)*r is for "pointy top" where horizontal spacing is 2?
    # No, let's use the standard axial to pixel for "pointy top":
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # Here size=1 (side length). So adjacent distance is 2*size? No, adjacent distance is sqrt(3)*size.
    # Wait, for a regular hexagon of side 1, the distance between centers of adjacent hexagons in a tiling is sqrt(3).
    # Let's verify. A hexagon of side 1 has width sqrt(3) (distance between parallel sides) and height 2 (distance between opposite vertices).
    # In a tiling, if we place them side by side, the centers are sqrt(3) apart.
    # Let's use axial coordinates (q, r) for a "pointy top" hex grid.
    # Pixel x = sqrt(3) * (q + r/2)
    # Pixel y = 3/2 * r
    # Distance between (0,0) and (1,0) is sqrt(3). Correct.
    # Distance between (0,0) and (0,1) is 1.5? No, sqrt(0 + 2.25) = 1.5. But adjacent hexagons should be sqrt(3) apart.
    # Let's check (1, -1). x = sqrt(3)*(1-0.5) = sqrt(3)/2. y = 3/2*(-1) = -1.5. dist = sqrt(3/4 + 2.25) = sqrt(3). Correct.
    # So the neighbors of (0,0) are (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1).
    # Their distances are all sqrt(3).
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate hex ring positions
    def hex_ring_positions(ring):
        if ring == 0:
            return [(0, 0)]
        positions = []
        # Start at (ring, 0) and go around
        q, r = ring, 0
        for side in range(6):
            for step in range(ring):
                positions.append((q, r))
                dq = [1, 0, -1, -1, 0, 1][side]
                dr = [0, -1, -1, 0, 1, 1][side]
                q += dq
                r += dr
        return positions

    # Collect positions by ring to form a hexagonal cluster
    all_positions = [(0, 0)]
    ring = 1
    while len(all_positions) < N:
        all_positions.extend(hex_ring_positions(ring))
        ring += 1
    
    selected_axial = all_positions[:N]
    
    # Convert to Cartesian
    inner_cart = []
    for q, r in selected_axial:
        x, y = axial_to_cartesian(q, r)
        inner_cart.append((x, y))
    
    # Centroid
    cx_mean = sum(p[0] for p in inner_cart) / len(inner_cart)
    cy_mean = sum(p[1] for p in inner_cart) / len(inner_cart)
    
    # Center the cluster at origin
    centered_inner = [(x - cx_mean, y - cy_mean) for x, y in inner_cart]
    
    # We want to find the minimum side length S of a regular hexagon (outer)
    # such that all inner unit hexagons (side 1, angle 0) are inside.
    # The inner hexagons are fixed in orientation (angle 0).
    # The outer hexagon can be rotated.
    # Let the outer hexagon have center (0,0), side S, and rotation angle theta.
    
    # A point (px, py) is inside a hexagon centered at (0,0) with side S and rotation theta
    # if and only if for all 6 directions, the projection of the point onto the normal
    # is <= apothem.
    # Apothem = S * sqrt(3) / 2.
    # The normals are at angles theta + 30 + 60*k.
    # Wait, the vertices of a hexagon with side S and rotation theta are at angles theta + 60*k.
    # The normals to the sides are at angles theta + 30 + 60*k.
    
    def is_point_in_hexagon(px, py, S, theta):
        apothem = S * sqrt3 / 2.0
        for k in range(6):
            angle = theta + math.pi / 6 + k * math.pi / 3  # 30 degrees + 60k
            nx = math.cos(angle)
            ny = math.sin(angle)
            if px * nx + py * ny > apothem + 1e-9:
                return False
        return True
    
    def is_inner_hex_in_outer(ix, iy, S, theta):
        # Inner hexagon has side 1, angle 0.
        # Its vertices are at (cos(60k), sin(60k)) + (ix, iy).
        for k in range(6):
            vx = ix + math.cos(k * math.pi / 3)
            vy = iy + math.sin(k * math.pi / 3)
            if not is_point_in_hexagon(vx, vy, S, theta):
                return False
        return True

    def min_side_for_theta(theta, inner_list):
        # Binary search for minimum S
        lo = 0.0
        hi = 50.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            valid = True
            for ix, iy in inner_list:
                if not is_inner_hex_in_outer(ix, iy, mid, theta):
                    valid = False
                    break
            if valid:
                hi = mid
            else:
                lo = mid
        return hi + 1e-10

    # We should try different rotations of the outer hexagon.
    # Also, we can translate the inner cluster slightly to optimize?
    # The problem allows translating inner hexagons.
    # But for now, let's just center them and try outer rotations.
    # Actually, the inner cluster shape is fixed relative to itself.
    # We can rotate the entire inner cluster? No, the inner hexagons can rotate individually.
    # But the optimal packing usually has them aligned.
    # Let's assume inner hexagons are all aligned (angle 0).
    
    best_side = float('inf')
    best_theta = 0.0
    
    # Try a range of outer rotations
    # Symmetry of hexagon is 60 degrees.
    # So we only need to check 0 to 60 degrees.
    # Let's sample finely.
    
    if time.time() < deadline - 10:
        # Coarse search
        for theta_deg in range(0, 61, 5):
            theta = math.radians(theta_deg)
            side = min_side_for_theta(theta, centered_inner)
            if side < best_side:
                best_side = side
                best_theta = theta
        
        # Fine search around best
        if best_theta > 0:
            start_deg = int(math.degrees(best_theta)) - 5
            end_deg = int(math.degrees(best_theta)) + 5
            for theta_deg_10 in range(start_deg * 10, end_deg * 10 + 1, 1):
                theta = math.radians(theta_deg_10 / 10.0)
                side = min_side_for_theta(theta, centered_inner)
                if side < best_side:
                    best_side = side
                    best_theta = theta
    
    # Construct the final result
    # Inner: (x, y, angle_degrees)
    inner = [[x, y, 0.0] for x, y in centered_inner]
    outer_center = [0.0, 0.0]
    outer_side = best_side
    outer_angle = math.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
