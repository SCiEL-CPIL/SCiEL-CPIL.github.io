import numpy as np

def hexagon_packing_11():
    # Define the centers of the 11 inner hexagons
    # Using a 3-4-3 arrangement on a honeycomb lattice
    # Lattice spacing is sqrt(3)
    s3 = np.sqrt(3)
    
    centers = []
    # Middle row (4 hexagons)
    for i in range(4):
        x = (i - 1.5) * s3
        y = 0
        centers.append([x, y])
        
    # Top row (3 hexagons)
    for i in range(3):
        x = (i - 1) * s3
        y = 1.5
        centers.append([x, y])
        
    # Bottom row (3 hexagons)
    for i in range(3):
        x = (i - 1) * s3
        y = -1.5
        centers.append([x, y])
        
    centers = np.array(centers)
    
    # Inner hexagons are "flat-top" (angle 30 degrees)
    # This aligns them with the lattice vectors for dense packing
    inner_hex_data = np.column_stack((centers[:, 0], centers[:, 1], np.full(11, 30.0)))
    
    # Outer hexagon
    # Optimal orientation for this cluster is "pointy-top" (angle 0)
    # Centered at origin
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Calculate minimal side length S
    # Extent in X: 2 * sqrt(3)
    # Extent in Y: 2.5
    # Constraint 1: S >= 2 * sqrt(3)
    # Constraint 2: S * sqrt(3) / 2 >= 2.5  => S >= 5 / sqrt(3)
    S_x = 2 * s3
    S_y = 5 / s3
    
    outer_hex_side_length = max(S_x, S_y)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
