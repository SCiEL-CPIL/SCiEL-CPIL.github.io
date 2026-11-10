# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using a hexagonal lattice cluster (radius 2) and computing the minimal containing regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), rows (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Hexagonal lattice of 12 unit hexagons: 1 center + 6 ring + 5 outer
    # Centers on triangular lattice with spacing sqrt(3)
    # Use axial coordinates for hexagonal grid, convert to cartesian
    # For hexagons with flat-top orientation (angle=0), neighbors are at:
    #   (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5),
    #   (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    
    sqrt3 = np.sqrt(3)
    
    # Axial coordinates (q, r) for the 12 hexagons in a hex cluster of radius 2
    # Center: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: we need 5 more to total 12 (a full ring 2 would be 12, but we only need 5)
    # Actually, for a compact shape, use the standard hex number: 1+6+12=19 for radius 2
    # But we only have 12. Let's use: center(1) + ring1(6) + 5 from ring2
    
    # Better approach: use the 12 positions that form the most compact shape
    # A hexagon of hexagons with "radius" such that we have 12 total
    # The compact arrangement: think of it as a 3x4 rectangular-ish block in hex coords
    
    # Let me use explicit positions. For a flat-top hexagon grid:
    # Position (q, r) in axial coords maps to:
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    
    # 12 hexagons in a compact hexagonal shape:
    # Use the 12 positions closest to origin in the hex lattice
    # These form a hexagon of side 2 (which has 1+6+12=19... no)
    
    # Actually, the hex number H(n) = 3n(n-1)+1
    # H(1)=1, H(2)=7, H(3)=19
    # So 12 is between H(2)=7 and H(3)=19
    # The most compact 12-hexagon shape: take the 7 from radius 1, plus 5 from radius 2
    
    # Let me just define the 12 centers explicitly for a compact arrangement
    # Using a "flower" + partial outer ring
    
    centers = []
    # Center
    centers.append((0, 0))
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
    
    # Ring 2: add 5 hexagons at distance 2*sqrt(3)
    # Place them at angles that fill gaps: 0, 60, 120, 180, 240 (skip 300 for compactness)
    # Actually, to be symmetric, let's place them at 30, 90, 150, 210, 270, 330 minus one
    # For maximum compactness, place 5 at: 0, 72, 144, 216, 288? No, that's not hex lattice.
    
    # In hex lattice, ring 2 positions are at distance 2*sqrt(3) from center
    # at angles: 0, 60, 120, 180, 240, 300 (6 positions) plus 6 more at 30, 90, ...
    # Total 12 positions in ring 2. We pick 5.
    
    # For compactness, pick 5 that minimize the bounding hexagon
    # Let's try: 0, 60, 120, 180, 240 (5 consecutive)
    for i in range(5):
        angle = i * np.pi / 3
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # All inner hexagons have angle 0 (flat-top)
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Now compute the minimal containing regular hexagon
    # A regular hexagon with center (cx, cy), side s, angle a has vertices at:
    # (cx + s*cos(a + 60*i*deg), cy + s*sin(a + 60*i*deg))
    # We want to find minimal s such that all inner hexagon vertices are inside.
    
    # By symmetry, optimal outer hexagon is centered at origin with angle 0
    # Compute all vertices of all 12 inner hexagons
    all_vertices = []
    for i in range(12):
        cx, cy, angle_deg = inner_hex_data[i]
        angle_rad = np.deg2rad(angle_deg)
        for j in range(6):
            vx = cx + np.cos(angle_rad + j * np.pi / 3)
            vy = cy + np.sin(angle_rad + j * np.pi / 3)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # For a regular hexagon centered at origin with angle 0, the 6 edge normals are at angles:
    # 0, 60, 120, 180, 240, 300 degrees
    # The hexagon contains point (x,y) iff for all 6 directions, the projection is <= s
    # Actually, for a regular hexagon with side s, the apothem (distance from center to edge) is s*sqrt(3)/2
    # The condition is: for each of 6 edge normals n_i, dot(v, n_i) <= s*sqrt(3)/2
    
    # Edge normals for angle=0 hexagon:
    normals = []
    for i in range(6):
        angle = i * np.pi / 3
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals)
    
    # For each vertex, compute max projection onto each normal
    # The required s must satisfy: s * sqrt(3)/2 >= max_i dot(v, n_i) for all vertices v
    # So s >= (2/sqrt(3)) * max_over_all_vertices_and_normals(dot(v, n_i))
    
    # Compute all dot products
    # all_vertices: (72, 2), normals: (6, 2)
    projections = all_vertices @ normals.T  # (72, 6)
    max_proj = np.max(projections)
    
    # s * sqrt(3)/2 >= max_proj
    s = 2 * max_proj / sqrt3
    
    # Center outer hexagon at origin, angle 0
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
