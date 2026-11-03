"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place N=11 unit regular hexagons in a compact hexagonal lattice arrangement.
    # Use axial coordinates (q, r) for a pointy-top hexagonal grid.
    # Spacing: dx = sqrt(3), dy = 1.5 for adjacent hexes.
    # For a unit hexagon (side 1), the vertices in pointy-top orientation are at:
    #   angles 30, 90, 150, 210, 270, 330 degrees, radius = 1 (circumradius = side for regular hexagon)
    
    # Compact arrangement: center + 6 neighbors + 4 additional = 11
    # Axial coords: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)  -- 7 hexes (radius 1)
    # Add: (2,0), (-2,0), (0,2), (0,-2)  -- 4 more = 11 total
    
    axial_positions = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    # Convert axial to Cartesian for pointy-top hexagons
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    sqrt3 = math.sqrt(3)
    
    inner = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Now compute the minimum side length S of a regular hexagon (pointy-top, 0-degree rotation)
    # centered at (cx, cy) that contains all inner hexagons.
    # 
    # A pointy-top regular hexagon with center (cx, cy) and side S has the following half-plane constraints:
    # The hexagon is the intersection of 6 half-planes. For pointy-top orientation:
    # The 6 normals point at angles 0, 60, 120, 180, 240, 300 degrees.
    # Wait, let me be more careful.
    #
    # A regular hexagon with circumradius R = S (side length = circumradius for regular hexagon).
    # For pointy-top orientation, the vertices are at angles 90, 150, 210, 270, 330, 30 degrees from center.
    # The 6 edges have normals at angles 0, 60, 120, 180, 240, 300 degrees.
    # Each edge is at distance R * cos(30°) = R * sqrt(3)/2 from center along the normal direction.
    #
    # So constraint for each normal angle theta: 
    #   (px - cx) * cos(theta) + (py - cy) * sin(theta) <= R * sqrt(3)/2
    #
    # But we also need to contain the entire inner hexagons, not just their centers.
    # Each inner hexagon has vertices at distance 1 from its center.
    # So for each inner hexagon at (ix, iy), we need to check its 6 vertices.
    #
    # Actually, to be safe, we need the outer hexagon to contain all points of all inner hexagons.
    # The farthest point of an inner hexagon from any direction is at distance 1 from its center.
    # So for a given direction (normal angle theta), the max extent of an inner hexagon is:
    #   (ix - cx)*cos(theta) + (iy - cy)*sin(theta) + 1  (approximately, since max projection of unit hexagon on any axis is 1)
    #
    # Wait, the max projection of a unit regular hexagon onto any line is actually its width in that direction.
    # For a pointy-top hexagon, the width along the x-axis is sqrt(3), along y-axis is 2.
    # The max projection onto a direction at angle theta is:
    #   max over vertices of (vx * cos(theta) + vy * sin(theta))
    # For a unit hexagon centered at origin with pointy-top:
    #   vertices at angles 30, 90, 150, 210, 270, 330, radius 1
    #   v_k = (cos(30+60k), sin(30+60k)) for k=0..5
    #   max projection = max_k (cos(30+60k)*cos(theta) + sin(30+60k)*sin(theta))
    #                 = max_k cos((30+60k) - theta)
    #                 = max over 6 angles of cos(angle_diff)
    #   The max of cos over 6 evenly spaced angles (every 60 deg) starting at 30 deg offset is:
    #   The closest angle to 0 (mod 360) among {30, 90, 150, 210, 270, 330} relative to theta.
    #   The minimum angular distance to any of these is at most 30 degrees.
    #   So max projection = cos(30°) = sqrt(3)/2? No wait.
    #
    # Let me just compute the 6 vertex projections directly for each inner hexagon.
    
    # For pointy-top hexagon, vertices at angles: 30, 90, 150, 210, 270, 330 degrees
    vertex_angles = [math.radians(30 + 60 * k) for k in range(6)]
    
    # For each of the 6 constraint directions (normals at 0, 60, 120, 180, 240, 300 degrees)
    # compute the required S.
    
    # The outer hexagon must satisfy for each normal direction theta_j (j=0..5):
    #   max over all inner hexagons i, all vertices v of (px + vx, py + vy) projected onto (cos theta_j, sin theta_j)
    #   <= cx * cos(theta_j) + cy * sin(theta_j) + S * sqrt(3)/2
    #
    # If we center the outer hexagon at the centroid of all inner hexagon vertices, we can solve for S.
    # Actually, for a given center, the minimum S is:
    #   S = max_j [ (max_proj_j - cx*cos(theta_j) - cy*sin(theta_j)) / (sqrt(3)/2) ]
    #
    # To minimize S, we should choose (cx, cy) optimally. This is a linear programming problem,
    # but for simplicity, let's try centering at the centroid of all inner hexagon centers first.
    
    # Centroid of inner hexagon centers
    cx = sum(p[0] for p in inner) / N
    cy = sum(p[1] for p in inner) / N
    
    # Normal directions for pointy-top outer hexagon
    normal_angles = [math.radians(60 * j) for j in range(6)]
    
    # For each constraint direction, find the max projection of all vertices of all inner hexagons
    S_required = 0.0
    for theta in normal_angles:
        ct = math.cos(theta)
        st = math.sin(theta)
        max_proj = float('-inf')
        for ix, iy, _ in inner:
            for va in vertex_angles:
                vx = math.cos(va)
                vy = math.sin(va)
                px = ix + vx
                py = iy + vy
                proj = px * ct + py * st
                if proj > max_proj:
                    max_proj = proj
        # The constraint: max_proj <= cx*ct + cy*st + S * sqrt(3)/2
        # So S >= (max_proj - cx*ct - cy*st) / (sqrt(3)/2)
        S_needed = (max_proj - cx * ct - cy * st) / (sqrt3 / 2)
        if S_needed > S_required:
            S_required = S_needed
    
    side = S_required
    return inner, (cx, cy), side, 0.0
# EVOLVE-BLOCK-END
