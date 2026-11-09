"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Find hexagonal cluster radius r such that 3r(r+1)+1 >= N
    r = 0
    while 3 * r * (r + 1) + 1 < N:
        r += 1
    
    # Generate hexagonal lattice positions for unit hexagons (flat-top)
    # Center-to-center distance for flat-top hexagons: dx = 3/2 * side, dy = sqrt(3)/2 * side
    # For side=1: adjacent centers are sqrt(3) apart in the "column" direction
    # Actually for flat-top hexagons, the horizontal spacing between columns is 3/2, 
    # and vertical spacing is sqrt(3), with odd columns offset by sqrt(3)/2
    
    # Let's use axial coordinates for hexagonal grid
    # For a hexagon of radius r, we include all (q, q, q) with |q|+|q|+|q| <= r in cube coords
    positions = []
    for q in range(-r, r + 1):
        for q2 in range(max(-r, -q - r), min(r, -q + r) + 1):
            q3 = -q - q2
            # Convert to pixel coordinates for flat-top hexagons
            # x = size * 3/2 * q
            # y = size * sqrt(3) * (q2 + q/2)
            x = 1.5 * q
            y = np.sqrt(3) * (q2 + q / 2.0)
            positions.append((x, y))
    
    # We may have more positions than N, take the first N
    positions = positions[:N]
    
    if len(positions) == 0:
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.0, 0.0
    
    # Compute the bounding regular hexagon
    # For a regular hexagon centered at origin with side length s and rotation angle theta,
    # the constraint is that all points must satisfy: for each of 6 directions, the projection <= s*sqrt(3)/2
    # Actually, the apothem (distance from center to side) of a regular hexagon with side s is s*sqrt(3)/2
    
    # We need to find the smallest s such that all unit hexagons fit inside.
    # A unit hexagon at position (px, py) with rotation 0 has vertices at distance 1 from its center.
    # So the effective "radius" we need to contain is 1 (circumradius of unit hexagon).
    
    # For the outer hexagon with side s, apothem = s * sqrt(3)/2
    # The outer hexagon contains a point at distance d from center if d <= s * sqrt(3)/2 in the normal direction...
    # Actually for a regular hexagon, the maximum distance from center to boundary varies with direction.
    # In the direction of a vertex: distance = s
    # In the direction of a side midpoint: distance = s * sqrt(3)/2
    
    # To be safe, we need the outer hexagon's inradius (apothem) >= max over all points of (distance from center + 1)
    # But that's not quite right either since the unit hexagon has extent in all directions.
    
    # Simpler approach: the outer hexagon must contain all vertices of all inner hexagons.
    # Each inner hexagon at (px, py) has 6 vertices at distance 1 from (px, py).
    # So we need the outer hexagon to contain all points at distance <= 1 from each (px, py).
    
    # The outer regular hexagon with center (0,0), side s, rotation 0 (flat-top) has the property:
    # A point (x,y) is inside if for all 6 normals n_i, dot((x,y), n_i) <= s*sqrt(3)/2
    # where n_i are unit vectors at angles 0, 60, 120, 180, 240, 300 degrees... 
    # Actually for flat-top hexagon, the sides are at angles 0, 60, 120, 180, 240, 300 from horizontal.
    # The normals point outward at those angles.
    
    # Let's just compute: for each inner hexagon center, the farthest vertex in each of the 6 normal directions.
    # Then s must satisfy: max_over_hexagons, max_over_vertices dot(vertex, n_i) <= s * sqrt(3)/2
    
    # Vertices of a flat-top unit hexagon centered at (px, py):
    # (px + cos(k*pi/3), py + sin(k*pi/3)) for k=0..5
    
    # Normals for outer hexagon (flat-top, rotation 0):
    # The sides are horizontal and at ±60 degrees. Normals point at 0, 60, 120, 180, 240, 300 degrees.
    normal_angles = [k * np.pi / 3 for k in range(6)]
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    
    max_proj = 0.0
    for (px, py) in positions:
        for k in range(6):
            vx = px + np.cos(k * np.pi / 3)
            vy = py + np.sin(k * np.pi / 3)
            for n in normals:
                proj = vx * n[0] + vy * n[1]
                if proj > max_proj:
                    max_proj = proj
    
    # apothem = s * sqrt(3)/2 >= max_proj
    s = max_proj / (np.sqrt(3) / 2)
    
    inner = [[px, py, 0.0] for (px, py) in positions]
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END
