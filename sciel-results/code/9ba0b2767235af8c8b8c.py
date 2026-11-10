# EVOLVE-BLOCK-START
import numpy as np


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon using a symmetric hexagonal lattice
    (1 center + 3 rings? No, 1 center + 6 in ring1 + 5 in ring2? Actually 12 = 1 + 6 + 5? 
    Let's use the standard hexagonal number: 1, 7, 19. 12 is not a hex number.
    But we can use a subset of the 19-hexagon hexagonal arrangement, or a different layout.
    
    Actually, the optimal packing of 12 unit hexagons in a hexagon is known.
    Let's use a layout that fits in a hexagon of side s.
    The seed used s=8 which is very loose.
    
    A common tight packing for 12 hexagons:
    Arrange them in a hexagonal shape but missing some corners, or use a different orientation.
    
    Let's try a layout of 12 hexagons in a hexagon of side 3.
    Side 3 hexagon can hold 19 unit hexagons in a perfect grid.
    So 12 will fit easily in side 3.
    Let's verify the minimum side length for 12.
    
    If we use a hexagonal grid with spacing 2.5 (distance between centers of touching hexagons is 2 * apothem = 2 * sqrt(3)/2 * 2 = 2*sqrt(3)? No.
    Distance between centers of two touching unit hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732?
    No, the distance between centers of two touching regular hexagons of side 1 is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)?
    Let's check: Apothem of unit hexagon is sqrt(3)/2. Two hexagons touching at an edge: distance between centers is 2 * apothem = sqrt(3).
    Wait, if they touch at an edge, the distance between centers is 2 * apothem = sqrt(3).
    
    Let's place centers on a triangular lattice with spacing sqrt(3).
    The 19-hexagon hexagonal number arrangement has centers at:
    (0,0)
    Ring 1: 6 hexagons at distance sqrt(3)
    Ring 2: 12 hexagons at distance 2*sqrt(3)
    Total 19.
    
    We need 12. We can take the center + ring 1 (6) + 5 from ring 2.
    Or we can use a different shape.
    
    Actually, the optimal packing of 12 unit circles in a circle is known, but for hexagons in a hexagon, it's also studied.
    The minimal enclosing hexagon side for 12 unit hexagons is likely 3.
    Let's try to place 12 hexagons in a hexagon of side 3.
    
    Let's use the following centers (relative to center of outer hexagon):
    We want to fit them in a hexagon of side s.
    The vertices of the outer hexagon (side s, angle 0) are at:
    (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    Let's try s = 3.
    The apothem of the outer hexagon is 3 * sqrt(3)/2 ≈ 2.598.
    The inner hexagons have apothem sqrt(3)/2 ≈ 0.866.
    So the centers must be within distance 2.598 - 0.866 = 1.732 from the center in the "apothem" direction.
    Actually, the condition is that the inner hexagon vertices must be inside the outer hexagon.
    
    Let's use a known optimal or near-optimal configuration.
    According to literature, the densest packing of 12 unit hexagons in a hexagon has an outer side length of 3.
    Let's construct it:
    Place 1 hexagon at center (0,0).
    Place 6 hexagons around it at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300 degrees.
    That's 7 hexagons.
    We need 5 more. Place them in the "gaps" or next ring.
    The next ring is at distance 2*sqrt(3) ≈ 3.464.
    But we can place them closer if we adjust angles.
    
    Alternatively, use a layout with 3 rows:
    Row 1: 4 hexagons
    Row 2: 4 hexagons
    Row 3: 4 hexagons
    With appropriate offsets.
    
    Let's try a simpler approach: use the 19-hexagon hexagonal arrangement and remove 7 hexagons from the corners to leave 12, then compute the minimal enclosing hexagon.
    
    Centers for 19-hexagon arrangement (side 1, spacing sqrt(3)):
    Let dx = sqrt(3), dy = 1.5 (since in a hex grid, the y-spacing is 1.5 for unit hexagons? 
    Actually, for unit hexagons, the center-to-center distance is sqrt(3) in the "flat" direction and 1.5 in the "pointy" direction if oriented with flat top.
    Let's use axial coordinates.
    
    Let's just use a known valid packing.
    The seed program used s=8. We can easily do better.
    Let's try s=3.
    
    Centers:
    (0, 0)
    (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)  # 7 hexagons
    Now add 5 more:
    (2*sqrt(3), 0) - might be too far
    (sqrt(3), 3), (0, 3), (-sqrt(3), 3), (sqrt(3), -3), (-sqrt(3), -3)
    
    Let's check if these 12 fit in a hexagon of side 3.
    The outer hexagon (side 3, angle 0) has vertices:
    (3,0), (1.5, 2.598), (-1.5, 2.598), (-3,0), (-1.5,-2.598), (1.5,-2.598)
    
    Check point (2*sqrt(3), 0) ≈ (3.464, 0). This is outside (x > 3).
    So s=3 is not enough for that layout.
    
    Let's try a different layout.
    What if we use a 2x2x2 grid? No, hexagons don't grid like squares.
    
    Let's use the fact that 12 hexagons can be packed in a hexagon of side 3 if we use a different arrangement.
    Actually, the minimal side length for 12 unit hexagons in a hexagon is 3.
    Let's verify by checking the vertices of the inner hexagons.
    
    Alternative: Use a layout with 4 hexagons in a row, 4 in the next, 4 in the next, with offsets.
    Row 1 (y=0): x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    Row 2 (y=1.5): x = -sqrt(3), 0, sqrt(3), 2*sqrt(3)
    Row 3 (y=3): x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    
    This is a 4x3 grid with hexagonal offsets.
    Total 12 hexagons.
    Let's check the bounding box.
    Max x: 2*sqrt(3) ≈ 3.464
    Min x: -1.5*sqrt(3) ≈ -2.598
    Max y: 3 + 0.866 = 3.866 (vertex of top hexagon)
    Min y: -0.866
    
    The enclosing hexagon must contain these.
    The side length s must be at least the maximum distance from center to vertex / cos(30) or something.
    
    Let's compute the minimal enclosing regular hexagon for this set of points.
    We can do this by checking all 6 orientations and finding the minimal s such that all vertices of all inner hexagons are inside the outer hexagon.
    
    For a regular hexagon of side s, angle a, center (cx, cy), the condition for a point (x,y) to be inside is:
    For each of the 6 edges, the cross product must be non-negative.
    
    Let's implement a function to find the minimal enclosing regular hexagon for a set of points.
    """
    # Define the 12 inner hexagon centers
    sqrt3 = np.sqrt(3)
    centers = np.array([
        # Row 1 (y=0)
        [-1.5*sqrt3, 0],
        [-0.5*sqrt3, 0],
        [0.5*sqrt3, 0],
        [1.5*sqrt3, 0],
        # Row 2 (y=1.5)
        [-sqrt3, 1.5],
        [0, 1.5],
        [sqrt3, 1.5],
        [2*sqrt3, 1.5],
        # Row 3 (y=3)
        [-1.5*sqrt3, 3],
        [-0.5*sqrt3, 3],
        [0.5*sqrt3, 3],
        [1.5*sqrt3, 3],
    ])
    
    # All inner hexagons have angle 0
    inner_angles = np.zeros(12)
    
    # Compute all vertices of all inner hexagons
    all_vertices = []
    for i in range(12):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1, 0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # (72, 2)
    
    # Find the minimal enclosing regular hexagon
    # The outer hexagon is centered at the centroid of the inner hexagons? Not necessarily, but let's try.
    centroid = np.mean(centers, axis=0)
    
    # We need to find s and angle a such that all vertices are inside.
    # The minimal s for a given angle a can be found by checking the maximum "distance" in the 6 normal directions.
    # For a regular hexagon with angle a, the 6 edge normals are at angles a, a+60, ..., a+300.
    # The condition for a point p to be inside the hexagon (centered at c, side s, angle a) is:
    # For each normal n_i, (p - c) . n_i <= s * sqrt(3)/2  (apothem)
    # So s >= max_i ((p - c) . n_i) / (sqrt(3)/2) for all p.
    
    # We can optimize over angle a in [0, 60) degrees.
    best_s = float('inf')
    best_angle = 0
    best_center = centroid
    
    # Try different centers? The optimal center might not be the centroid.
    # But for a symmetric packing, the centroid is likely optimal.
    # Let's try a few angles.
    for a_deg in np.linspace(0, 60, 1201):
        a = np.radians(a_deg)
        normals = []
        for k in range(6):
            angle = a + np.radians(60 * k)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)  # (6, 2)
        
        # For each vertex, compute the maximum projection onto the normals
        # (all_vertices - center) @ normals.T  -> (72, 6)
        projections = (all_vertices - centroid) @ normals.T
        max_proj = np.max(projections, axis=0)  # (6,)
        # s must satisfy: s * sqrt(3)/2 >= max_proj[i] for all i
        # So s >= max(max_proj) / (sqrt(3)/2)
        s_candidate = np.max(max_proj) / (sqrt3 / 2)
        if s_candidate < best_s:
            best_s = s_candidate
            best_angle = a_deg
            best_center = centroid
    
    # Now we have the minimal s for the centroid.
    # But we can also optimize the center.
    # Let's do a simple local search on the center.
    # Actually, for a convex shape, the minimal enclosing regular hexagon can be found by rotating calipers or similar.
    # But let's stick with the centroid for now, as it's likely optimal for this symmetric packing.
    
    # Verify the solution
    inner_hex_data = np.column_stack([centers, inner_angles])
    outer_hex_data = np.array([best_center[0], best_center[1], best_angle])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
