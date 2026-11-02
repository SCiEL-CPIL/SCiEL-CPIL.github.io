import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 centers of the unit hexagons in a compact 3x4 triangular lattice block
    # Distance between centers is sqrt(3)
    d = np.sqrt(3)
    
    # Row 0 (y=0)
    x_coords_r0 = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    y_coords_r0 = [0.0, 0.0, 0.0, 0.0]
    
    # Row 1 (y=1.5)
    x_coords_r1 = [-1.0 * d, 0.0, 1.0 * d, 2.0 * d]
    y_coords_r1 = [1.5, 1.5, 1.5, 1.5]
    
    # Row 2 (y=3.0)
    x_coords_r2 = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    y_coords_r2 = [3.0, 3.0, 3.0, 3.0]
    
    # Combine and center the cluster at (0,0)
    xs = np.concatenate([x_coords_r0, x_coords_r1, x_coords_r2])
    ys = np.concatenate([y_coords_r0, y_coords_r1, y_coords_r2])
    
    # Center the points
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs = xs - cx
    ys = ys - cy
    
    points = np.column_stack((xs, ys))
    
    # Find the minimum side length S of the outer hexagon
    # The outer hexagon must contain the inner hexagons.
    # If inner hexagons are aligned with outer, the centers must be inside
    # a hexagon of side S_eff = S_out - 1.
    # So we find min S_eff for the points, then S_out = S_eff + 1.
    
    min_apothem = float('inf')
    best_angle = 0.0
    
    # Scan angle from 0 to 60 degrees
    # Use 1 degree resolution for speed and accuracy
    for angle_deg in np.arange(0, 60, 0.5):
        angle_rad = np.radians(angle_deg)
        
        # Normals for the 3 unique directions of the hexagon sides
        # A regular hexagon has 6 sides, but opposite sides are parallel.
        # We need to check 3 directions.
        # The normals are at angles: alpha, alpha + 60, alpha + 120
        # Wait, the apothem is the distance to the side.
        # The condition is |dot(point, normal)| <= apothem
        # for the 3 pairs of parallel sides.
        
        n1 = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        n2 = np.array([np.cos(angle_rad + np.pi/3), np.sin(angle_rad + np.pi/3)])
        n3 = np.array([np.cos(angle_rad + 2*np.pi/3), np.sin(angle_rad + 2*np.pi/3)])
        
        # Calculate projections of all points onto these normals
        proj1 = np.abs(points @ n1)
        proj2 = np.abs(points @ n2)
        proj3 = np.abs(points @ n3)
        
        # The required apothem is the max of all projections
        current_apothem = max(np.max(proj1), np.max(proj2), np.max(proj3))
        
        if current_apothem < min_apothem:
            min_apothem = current_apothem
            best_angle = angle_deg
            
    # Calculate outer side length
    # Apothem A = S_eff * sqrt(3) / 2  => S_eff = 2 * A / sqrt(3)
    S_eff = 2.0 * min_apothem / np.sqrt(3)
    S_out = S_eff + 1.0
    
    # Prepare inner hexagons list
    # All inner hexagons are aligned with angle 0.0
    inner = [[float(x), float(y), 0.0] for x, y in zip(xs, ys)]
    
    # Outer hexagon center is (0,0), side is S_out, angle is best_angle
    return inner, (0.0, 0.0), float(S_out), float(best_angle)
# EVOLVE-BLOCK-END
