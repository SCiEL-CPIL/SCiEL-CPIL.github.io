<<<<<<< SEARCH
    # 2. Convert axial to Cartesian
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
=======
    # 2. Convert axial to Cartesian
    # Use a slightly larger scale to ensure initial non-overlap, 
    # but we will optimize to fit tighter.
    # The "gap" between hexagons in a honeycomb is small.
    # We will initialize the outer ring with a 30 degree rotation to mimic optimal packing.
    
    centers = []
    angles = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
        # Check if it's the center hexagon (0,0)
        if q == 0 and r == 0:
            angles.append(0.0)
        else:
            # Outer ring gets 30 degrees rotation
            # This helps them interlock better
            angles.append(30.0)
            
    # Initial layout: Inner hexagons aligned, outer rotated
    initial_inner = [[x, y, a] for (x, y), a in zip(centers, angles)]
>>>>>>> REPLACE

