import numpy as np

def hexagon_packing_12():
    # Generate 12 hexagon centers in a compact cluster
    # Use a hexagonal grid
    # Start with 7 (side 2 cluster) and add 5
    
    # Lattice vectors
    d = np.sqrt(3)
    e1 = np.array([d, 0])
    e2 = np.array([d/2, 1.5])
    
    centers = []
    # Add center
    centers.append(np.array([0, 0]))
    
    # Add 6 neighbors (dist 1 in lattice)
    for i in range(6):
        angle = i * np.pi / 3
        # Neighbors are at distance sqrt(3)
        # Direction angles: 0, 60, 120, 180, 240, 300
        # But in this lattice, the neighbors are at specific vectors
        # v0 = e1
        # v1 = e2
        # v2 = e2 - e1
        # v3 = -e1
        # v4 = -e2
        # v5 = e1 - e2
        pass
    
    # Explicitly define the 7 cluster
    c0 = np.array([0,0])
    c1 = e1
    c2 = e2
    c3 = e2 - e1
    c4 = -e1
    c5 = -e2
    c6 = e1 - e2
    
    centers.extend([c1, c2, c3, c4, c5, c6])
    
    # We have 7. Need 5 more.
    # Add them to the "right" side to extend the shape.
    # c1 is at e1.
    # Next one at 2*e1
    centers.append(2*e1)
    centers.append(2*e1 + e2)
    centers.append(2*e1 + 2*e2)
    centers.append(2*e1 + 3*e2)
    centers.append(2*e1 + 4*e2)
    
    # We have 12 centers.
    # Now find the minimal bounding hexagon side s.
    # The outer hexagon is centered at the origin? 
    # It's better to center the outer hexagon at the centroid of the inner hexagons.
    centroid = np.mean(centers, axis=0)
    
    # Shift centers so centroid is at origin
    centers_shifted = centers - centroid
    
    # Calculate the required side length s.
    # For a hexagon aligned with the lattice, the condition is:
    # For all centers c, and all vertices v of unit hex, c+v is inside H_s.
    # This is equivalent to the max "hex distance" of the centers being <= s-1?
    # No, we need to check the projection.
    # The outer hexagon has normals at 0, 60, 120 deg.
    # The apothem is s * sqrt(3)/2.
    # The projection of a point p onto normal n must be <= apothem.
    # The furthest projection of the inner hexagon centered at c is 
    # proj(c, n) + max_proj_unit_hex(n).
    # max_proj_unit_hex(n) is the apothem of the unit hex = sqrt(3)/2.
    # So we need:
    # max_c |proj(c, n)| + sqrt(3)/2 <= s * sqrt(3)/2
    # |proj(c, n)| <= s * sqrt(3)/2 - sqrt(3)/2 = (s-1) * sqrt(3)/2
    # So s >= 1 + max_c ( |proj(c, n)| / (sqrt(3)/2) )
    
    max_s = 0
    normals = []
    for i in range(3):
        angle = i * np.pi / 3
        n = np.array([np.cos(angle), np.sin(angle)])
        normals.append(n)
        
    for c in centers_shifted:
        for n in normals:
            proj = np.dot(c, n)
            # We need |proj| + sqrt(3)/2 <= s * sqrt(3)/2
            # s >= (|proj| + sqrt(3)/2) / (sqrt(3)/2)
            # s >= |proj| / (sqrt(3)/2) + 1
            required_s = abs(proj) / (np.sqrt(3)/2) + 1
            if required_s > max_s:
                max_s = required_s
                
    s = max_s
    
    # Construct inner hex data
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0 # angle 0
        
    # Outer hex data
    outer_hex_data = np.array([centroid[0], centroid[1], 0])
    
    return inner_hex_data, outer_hex_data, s
