import numpy as np

def hexagon_packing_12():
    # Generate lattice points for a side-2 hexagon (19 points)
    # Axial coordinates (q, r)
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q + r) <= 2:
                points.append((q, r))
    
    # Convert to Cartesian coordinates for centers
    # e1 = (2, 0), e2 = (1, sqrt(3))
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in points:
        x = 2 * q + r
        y = sqrt3 * r
        centers.append((x, y))
    
    # Identify the 7 inner points (side-1 hexagon)
    # These are points with max(|q|, |r|, |q+r|) <= 1
    inner_points = []
    outer_points = []
    for p in points:
        q, r = p
        if max(abs(q), abs(r), abs(q+r)) <= 1:
            inner_points.append(p)
        else:
            outer_points.append(p)
            
    # inner_points has 7, outer_points has 12.
    # We need 12 total, so we need 5 from outer_points.
    # Strategy: Pick the 5 outer points that are closest to the origin (Euclidean distance of centers).
    # This avoids the "corners" which are farther away.
    
    outer_centers = []
    for p in outer_points:
        q, r = p
        x = 2 * q + r
        y = sqrt3 * r
        dist = np.sqrt(x*x + y*y)
        outer_centers.append((dist, p))
        
    # Sort by distance
    outer_centers.sort(key=lambda x: x[0])
    
    # Take the 5 closest
    selected_outer = [p for dist, p in outer_centers[:5]]
    
    # Combine
    selected_points = inner_points + selected_outer
    
    # Convert selected points to Cartesian centers
    final_centers = []
    for q, r in selected_points:
        x = 2 * q + r
        y = sqrt3 * r
        final_centers.append((x, y))
        
    # Now calculate the minimal containing hexagon side length S
    # The outer hexagon is centered at (0,0) with angle 0.
    # Normals are at 30, 90, 150, 210, 270, 330 degrees.
    # For each inner hexagon, we need to check its vertices.
    # Vertices of a unit hexagon at (cx, cy) with angle 0:
    # (cx + cos(60i), cy + sin(60i)) for i=0..5
    
    max_proj = 0.0
    normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    
    for cx, cy in final_centers:
        # Generate 6 vertices
        for i in range(6):
            angle = np.deg2rad(60 * i)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            
            # Project onto all 6 normals
            for na in normal_angles:
                proj = vx * np.cos(na) + vy * np.sin(na)
                # We want the maximum absolute projection?
                # The hexagon is centered at origin. The constraint is |proj| <= Apothem_outer.
                # Apothem_outer = S * sqrt(3) / 2.
                # So we need S * sqrt(3) / 2 >= max |proj|.
                if abs(proj) > max_proj:
                    max_proj = abs(proj)
                    
    # Calculate S
    # S * sqrt(3) / 2 = max_proj
    # S = 2 * max_proj / sqrt(3)
    S = 2 * max_proj / sqrt3
    
    # Construct return values
    # inner: (12, 3) array of (x, y, angle_deg)
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(final_centers):
        inner_hex_data[i, 0] = cx
        inner_hex_data[i, 1] = cy
        inner_hex_data[i, 2] = 0.0
        
    # outer: (3,) array (x, y, angle_deg)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, S

# EVOLVE-BLOCK-END
