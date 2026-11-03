import numpy as np

N = 12

def construct():
    # Define the 12 centers in a 3x4 hexagonal grid
    # i is column 0..2, j is row 0..3
    # Flat-top hexagon tiling
    # x = i * sqrt(3) + (j % 2) * sqrt(3)/2
    # y = j * 1.5
    
    sqrt3 = np.sqrt(3)
    centers = []
    for j in range(4):
        for i in range(3):
            x = i * sqrt3 + (j % 2) * sqrt3 / 2.0
            y = j * 1.5
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Translate to centroid
    centroid = np.mean(centers, axis=0)
    centers_centered = centers - centroid
    
    # Normals of the outer hexagon (flat-top, angle 0)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    min_s_avail = float('inf')
    best_angle = 0.0
    
    # Search for optimal rotation of the cluster
    # Due to symmetry, 0 to 30 degrees is sufficient
    # Step size 0.1 degrees is usually enough for high precision
    for deg in np.arange(0, 30, 0.1):
        theta = np.deg2rad(deg)
        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        rotated_centers = centers_centered @ R.T
        
        # Calculate max projection onto each normal
        # proj = rotated_centers @ normals.T
        # We need max over all points and all normals
        # proj shape: (12, 6)
        projs = rotated_centers @ normals.T
        max_proj = np.max(projs)
        
        # max_proj is A_req (apothem of the enclosing hexagon)
        # S_avail = 2 * A_req / sqrt(3)
        s_avail = 2.0 * max_proj / sqrt3
        
        if s_avail < min_s_avail:
            min_s_avail = s_avail
            best_angle = deg
            
    # Final outer side length
    # The centers must be inside a hexagon of side S_avail.
    # The inner hexagons (side 1) must be inside the outer hexagon.
    # If aligned, S_out = S_avail + 1.
    s_out = min_s_avail + 1.0
    
    # Construct the final positions
    # We need to rotate the centers back by best_angle and add centroid
    # Wait, the centers_centered were rotated by best_angle to fit.
    # So the actual positions relative to outer center (0,0) are:
    # R(best_angle) @ centers_centered + centroid?
    # No.
    # We found that if we rotate the cluster by `best_angle`, it fits in a fixed hexagon.
    # So the positions in the final frame (where outer hex is fixed) are the rotated ones.
    # But we shifted by centroid first.
    # So:
    # 1. Original centers C.
    # 2. Centered C' = C - Centroid.
    # 3. Rotated C'' = R(best_angle) @ C'.
    # 4. These C'' are the positions relative to the outer center (0,0).
    #    Because the outer hexagon is centered at 0,0.
    #    And the "fitting" was done against a hexagon centered at 0,0.
    
    R_best = np.array([[np.cos(np.deg2rad(best_angle)), -np.sin(np.deg2rad(best_angle))],
                       [np.sin(np.deg2rad(best_angle)), np.cos(np.deg2rad(best_angle))]])
    
    final_centers = centers_centered @ R_best.T
    
    # Create inner list
    inner = []
    for c in final_centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return inner, (0.0, 0.0), float(s_out), 0.0

# EVOLVE-BLOCK-END
