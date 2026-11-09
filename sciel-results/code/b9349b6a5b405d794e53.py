import numpy as np

def hexagon_packing_12():
    # Lattice basis vectors for unit hexagons (side 1)
    # Distance between centers is sqrt(3)
    # a = (sqrt(3), 0)
    # b = (sqrt(3)/2, 3/2)
    
    # We want 12 points.
    # 1 at (0,0)
    # 6 at ring 1
    # 5 at ring 2
    
    # Indices (i, j)
    points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1) # 5 from ring 2
    ]
    
    # Calculate Cartesian coordinates
    # Add small epsilon for separation
    eps = 2e-6 / np.sqrt(3)
    scale = 1.0 + eps
    
    coords = []
    for i, j in points:
        x = scale * (np.sqrt(3) * i + np.sqrt(3)/2 * j)
        y = scale * (1.5 * j)
        coords.append((x, y))
    
    # Calculate optimal side length S
    # D_max is distance of furthest point
    # We used points from ring 2, so D_max = 2*sqrt(3)*scale
    D_max = 2 * np.sqrt(3) * scale
    # Apothem required: D_max + sqrt(3)/2
    A = D_max + np.sqrt(3)/2
    S = 2 * A / np.sqrt(3)
    
    # Construct inner hex data
    inner_hex_data = np.zeros((12, 3))
    for idx, (x, y) in enumerate(coords):
        inner_hex_data[idx, 0] = x
        inner_hex_data[idx, 1] = y
        # Angle: flat side radial
        # Radial angle
        if x == 0 and y == 0:
            angle = 0
        else:
            radial_angle = np.arctan2(y, x)
            # We want a flat side normal to be at radial_angle
            # Vertices at a, a+60...
            # Flat side normals at a+30, a+90...
            # So a+30 = radial_angle => a = radial_angle - 30
            angle = np.degrees(radial_angle) - 30
        inner_hex_data[idx, 2] = angle
        
    # Outer hex data
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, S
