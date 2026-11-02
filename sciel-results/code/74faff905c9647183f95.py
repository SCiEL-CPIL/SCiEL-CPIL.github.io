import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the hexagonal lattice basis vectors for unit hexagons
    # Distance between centers of adjacent hexagons is sqrt(3)
    L1 = np.array([np.sqrt(3), 0.0])
    L2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate 3x4 grid of centers
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * L1 + j * L2
            centers.append(c)
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers -= centroid
    
    best_side = float('inf')
    best_theta = 0.0
    best_phi = 0.0
    best_inner = None
    
    # Search over cluster rotation theta and container rotation phi
    # Symmetry allows us to search 0 to 30 degrees for both
    theta_steps = np.linspace(0, np.pi/6, 61) # 0 to 30 deg
    phi_steps = np.linspace(0, np.pi/6, 61)   # 0 to 30 deg
    
    def support_function_hex(unit_hex_angle, direction):
        """
        Calculate the support function of a unit regular hexagon 
        rotated by unit_hex_angle, in the direction 'direction'.
        The hexagon has side length 1.
        """
        # The normals of the sides of a unit hexagon (flat top, normal at 0)
        # are at 0, 60, 120, 180, 240, 300 degrees.
        # The support function is (sqrt(3)/2) / cos(delta), 
        # where delta is the angle between the direction and the nearest normal.
        
        # Relative angle
        rel_angle = direction - unit_hex_angle
        # Map to [-pi/6, pi/6] relative to the nearest normal
        # Normals are spaced by pi/3
        # delta = (rel_angle % (pi/3)) - pi/6
        # But we need the absolute minimum angle to a normal.
        # Let's use the property: h(t) = (sqrt(3)/2) / cos( (t % pi/3) - pi/6 )
        # This formula works if we map t to the range [0, pi/3) and subtract pi/6?
        # If t=0, (0) - pi/6 = -pi/6. cos(-pi/6) = sqrt(3)/2. h = (sqrt(3)/2)/(sqrt(3)/2) = 1? 
        # Wait. At normal (t=0), distance is apothem sqrt(3)/2.
        # At vertex (t=pi/6), distance is 1.
        # Formula: h(t) = (sqrt(3)/2) / cos(t - pi/6) is wrong for t=0.
        
        # Correct derivation:
        # Side normal at 0. Distance to side is sqrt(3)/2.
        # Angle t. The distance from center to boundary along angle t.
        # If t is in [-pi/6, pi/6] relative to normal, r(t) = (sqrt(3)/2) / cos(t).
        # So we need to map rel_angle to the range [-pi/6, pi/6].
        
        rel = rel_angle % (np.pi/3)
        # Map [0, pi/3) to [-pi/6, pi/6)
        if rel > np.pi/6:
            rel -= np.pi/3
        # Now rel is in [-pi/6, pi/6]
        return (np.sqrt(3)/2) / np.cos(rel)

    for theta in theta_steps:
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_centers = centers @ rot_matrix.T
        
        for phi in phi_steps:
            # Normals of the outer hexagon
            # Outer hexagon rotated by phi.
            # Normals are at phi + k*pi/3
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + k * np.pi/3
                n_vec = np.array([np.cos(normal_angle), np.sin(normal_angle)])
                
                # Max projection of centers onto this normal
                projections = rotated_centers @ n_vec
                max_center_proj = np.max(projections)
                
                # Add the support of the inner hexagons in this direction
                # Inner hexagons are rotated by theta
                h_val = support_function_hex(theta, normal_angle)
                
                extent = max_center_proj + h_val
                if extent > max_extent:
                    max_extent = extent
            
            # The outer hexagon apothem A must be >= max_extent
            # A = S * sqrt(3) / 2  =>  S = 2 * A / sqrt(3)
            S = 2.0 * max_extent / np.sqrt(3)
            
            if S < best_side:
                best_side = S
                best_theta = theta
                best_phi = phi
                best_inner = rotated_centers

    # Construct the final inner list
    # The problem asks for (x, y, angle_degrees)
    # We assume the inner hexagons are rotated by best_theta
    angle_deg = np.degrees(best_theta)
    inner = []
    for c in best_inner:
        inner.append([c[0], c[1], angle_deg])
        
    # Outer hexagon center is (0,0), side is best_side, rotation is best_phi
    # Note: The evaluator might expect the outer hexagon to be axis-aligned or whatever minimizes score?
    # The score is the outer side length. The rotation of the outer hexagon doesn't change the side length.
    # We return the side length.
    
    return inner, (0.0, 0.0), best_side, best_phi
# EVOLVE-BLOCK-END
