# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Returns
        inner_hex_data: np.ndarray of shape (12,3), rows are (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) (x, y, angle_degrees).
        outer_hex_side_length: float.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin

    # 1. Define the initial configuration: a honeycomb cluster of 12 hexagons
    # Center at (0,0).
    # Neighbors at distance 2*sqrt(3) = ~3.464
    # Directions: 0, 60, 120, 180, 240, 300 degrees
    # Next ring: distance 4*sqrt(3) = ~6.928
    # Directions: 30, 90, 150, 210, 270, 330 degrees
    
    # Let's verify the "12 hexagons" shape.
    # A "flower" of 7 hexagons (1 center + 6 neighbors) has 7.
    # Adding 5 more to the outer ring?
    # Or a hexagon of side 2 has 1+6+12 = 19 hexagons. Too many.
    # A hexagon of side 1 has 1+6 = 7.
    # We need 12.
    # Shape: 7 (flower) + 5 attached to the outside?
    # Let's place 12 hexagons in a compact shape.
    # Coordinates for a hex grid (axial):
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)  -> 6 neighbors
    # Total 7.
    # Add 5 more:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) ?
    # Let's just use a specific set of 12 positions that are known to be compact.
    # Actually, the seed program used a specific arrangement.
    # Let's generate a honeycomb of 12.
    # Rows of hexagons:
    # Row -2: 3 hexagons
    # Row -1: 4 hexagons
    # Row 0: 3 hexagons
    # Row 1: 2 hexagons
    # Total 12.
    # This is a "hexagon" shape missing corners?
    
    # Let's use the coordinates from the seed but ensure they are valid, then optimize.
    # Seed coordinates:
    # [0, 0]
    # [-2.5, 0], [2.5, 0]
    # [-1.25, 2.17], [1.25, 2.17]
    # [-1.25, -2.17], [1.25, -2.17]
    # [-3.75, 2.17], [3.75, 2.17]
    # [-3.75, -2.17], [3.75, -2.17]
    # [0, -4]
    
    # This looks like a valid packing.
    # Let's start with this and optimize the outer hexagon size and position.
    # We can also rotate the whole cluster to minimize the bounding hexagon.
    
    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-1.25, 2.1650635094610965, 0.0],  # 2.165... is sqrt(3)/2 * 2.5? No.
        # Distance between centers in honeycomb is 2*sqrt(3) ~ 3.464.
        # Seed used 2.5 and 2.17.
        # 2.5^2 + 2.17^2 = 6.25 + 4.7089 = 10.9589. sqrt(10.9589) = 3.31.
        # This is less than 3.464. So the seed hexagons might overlap?
        # Wait, the seed score was 1.0, meaning it was valid.
        # Let's re-read the seed.
        # "Invalid: ... two inner hexagons not separated ... more than 1e-6 apart"
        # If the seed is valid, the distance must be sufficient.
        # For unit hexagons, the minimum distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732?
        # No, for edge-to-edge contact, distance is 2 * side * cos(30) * 2?
        # Apothem = d * sqrt(3)/2.
        # Distance between centers of touching hexagons = 2 * apothem = d * sqrt(3).
        # For d=1, distance = sqrt(3) ~ 1.732.
        # The seed uses distances like 2.5. 2.5 > 1.732. So they don't touch, they are separated.
        # This is a loose packing. We can tighten it!
        
        # Let's generate a proper honeycomb packing.
        # Center (0,0).
        # 6 neighbors at distance sqrt(3).
        # 12th hexagon?
        # A honeycomb of 12 hexagons is not a perfect hexagon.
        # Let's use a "brick wall" or similar compact shape.
        
        # Let's just use the seed positions as a starting point for optimization,
        # but I suspect we can pack them tighter.
        
        # Let's try to construct a tighter packing.
        # 1 center.
        # 6 around it. (Total 7)
        # 5 more attached to the outside.
        # Let's place them at (2, 0), (2, -sqrt(3)), (1, -2*sqrt(3)), (0, -2*sqrt(3)), (-1, -sqrt(3))?
        # Let's calculate coordinates for a hex grid with side 1.
        # x = 3/2 * q + sqrt(3)/2 * r
        # y = sqrt(3) * r
        # q, r are axial coordinates.
        
        # Let's pick 12 axial coordinates that form a compact shape.
        # (0,0)
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)  -> 6
        # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) -> 5
        # Let's check if this is valid (no overlaps).
        # In a hex grid, adjacent cells are at distance sqrt(3).
        # Non-adjacent cells are further.
        # So this set of 12 is a valid packing of 12 hexagons.
        
    ])
    
    # Let's generate the coordinates for the 12 hexagons in a hex grid.
    # Axial coordinates:
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1)
    ]
    
    inner_hex_data = np.zeros((12, 3))
    for i, (q, r) in enumerate(coords):
        x = 1.5 * q + np.sqrt(3) / 2 * r
        y = np.sqrt(3) * r
        inner_hex_data[i, 0] = x
        inner_hex_data[i, 1] = y
        inner_hex_data[i, 2] = 0.0  # angle

    # Now we need to find the smallest regular hexagon that contains all these.
    # The outer hexagon can be rotated.
    # We want to minimize the side length s of the outer hexagon.
    # The outer hexagon is centered at (cx, cy) with angle a.
    # We can fix the outer hexagon center at (0,0) by shifting the inner hexagons.
    # So we just need to find the rotation angle a that minimizes the required side length s.
    
    # For a given rotation a, the required side length s is determined by the
    # maximum projection of the inner hexagon vertices onto the 6 normals of the outer hexagon.
    
    def get_hex_vertices(cx, cy, d, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
        xs = cx + d * np.cos(angles)
        ys = cy + d * np.sin(angles)
        return np.column_stack((xs, ys))

    def min_enclosing_hexagon_side(inner_data):
        # inner_data: (12, 3) array of (x, y, angle)
        # We assume all inner hexagons have side 1.
        # We want to find the minimum s such that all inner hexagons fit inside
        # a regular hexagon of side s centered at (0,0) with some rotation.
        
        # The outer hexagon has 6 faces. The normal to face i is at angle a + i*60.
        # The distance from the center to the face is s * sqrt(3)/2 (apothem).
        # So for a point p to be inside, dot(p, normal_i) <= s * sqrt(3)/2 for all i.
        # So s >= max_i (dot(p, normal_i) / (sqrt(3)/2)) = max_i (2/sqrt(3) * dot(p, normal_i)).
        
        # We need to check all vertices of all inner hexagons.
        # There are 12 * 6 = 72 vertices.
        
        # Let's collect all vertices.
        vertices = []
        for i in range(12):
            x, y, ang = inner_data[i]
            verts = get_hex_vertices(x, y, 1.0, ang)
            vertices.append(verts)
        vertices = np.vstack(vertices)
        
        # Now we want to minimize s(a) = max_i (2/sqrt(3) * dot(p, normal_i(a)))
        # where normal_i(a) = (cos(a + i*60), sin(a + i*60)).
        
        # We can search over a in [0, 30) degrees.
        # Let's do a fine grid search.
        
        best_s = np.inf
        best_a = 0.0
        
        # Grid search for angle
        for a_deg in np.linspace(0, 30, 300):
            a_rad = np.deg2rad(a_deg)
            normals = []
            for i in range(6):
                theta = a_rad + i * np.pi / 3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            # Compute dot products for all vertices with all normals
            # vertices: (72, 2)
            # normals: (6, 2)
            dots = vertices @ normals.T  # (72, 6)
            
            max_dot = np.max(dots)
            s = max_dot * 2 / np.sqrt(3)
            
            if s < best_s:
                best_s = s
                best_a = a_deg
        
        # Refine the angle around best_a
        for a_deg in np.linspace(best_a - 1, best_a + 1, 100):
            if a_deg < 0: a_deg += 30
            a_rad = np.deg2rad(a_deg)
            normals = []
            for i in range(6):
                theta = a_rad + i * np.pi / 3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            dots = vertices @ normals.T
            max_dot = np.max(dots)
            s = max_dot * 2 / np.sqrt(3)
            
            if s < best_s:
                best_s = s
                best_a = a_deg
        
        return best_s, best_a

    # Calculate the initial best s and a
    s, a = min_enclosing_hexagon_side(inner_hex_data)
    
    # Now we can try to improve the packing by moving the inner hexagons.
    # Since we have time, we can try a simple local optimization.
    # But moving hexagons in a hex grid is tricky because they are constrained to not overlap.
    # A simpler approach: rotate the entire cluster and re-center.
    # We already did that in min_enclosing_hexagon_side.
    
    # Can we shrink the cluster?
    # The cluster is fixed by the hex grid coordinates.
    # We can try different shapes of 12 hexagons.
    
    # Let's try a few different shapes.
    # Shape 1: The one we used.
    # Shape 2: A 3x4 rectangle in hex grid?
    # Let's generate a few candidate shapes and pick the best one.
    
    shapes = []
    
    # Shape 1: The one from above
    shapes.append(coords)
    
    # Shape 2: A hexagon of side 2 missing 7 corners? No, side 2 has 19.
    # Let's try a "line" of 12? No, too long.
    # Let's try a "block" of 12.
    # 3 rows of 4?
    # In hex grid, a 3x4 block is not rectangular in Euclidean space.
    
    # Let's try to optimize the positions of the 12 hexagons freely, not just on a grid.
    # This is a hard problem.
    
    # Given the time constraint, let's just return the result from the grid shape.
    # We can improve it by allowing the inner hexagons to rotate slightly?
    # No, they are regular hexagons.
    
    # Let's check if we can improve the seed's score.
    # The seed used a loose packing. Our grid packing is tighter.
    # So our s should be smaller, and 1/s should be larger.
    
    # Let's calculate the value.
    value = 1.0 / s
    
    # Construct the return values.
    outer_hex_data = np.array([0.0, 0.0, a])
    outer_hex_side_length = s
    
    # We need to shift the inner hexagons so that the outer hexagon is centered at (0,0).
    # In our calculation, we assumed the outer hexagon is centered at (0,0).
    # The inner hexagons are at their original positions.
    # So we just need to return them as is.
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
