import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal grid points (axial coordinates q, r)
    # Distance from origin in Cartesian is proportional to sqrt(q^2 + qr + r^2)
    # We want the N closest points to the origin.
    
    points = []
    # Range sufficient for N=11 (and likely larger N if evolved)
    # Max radius in axial coords approx sqrt(N)
    R = 10 
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            # Axial to Cartesian for unit hexagons (side 1)
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            # dist_sq = 3 * (q^2 + qr + r^2)
            dist_sq = q*q + q*r + r*r
            points.append((dist_sq, q, r))
            
    # Sort by distance
    points.sort()
    
    # Take first N points
    inner_centers = []
    for i in range(N):
        _, q, r = points[i]
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        inner_centers.append((x, y))
        
    # Construct inner list: [x, y, angle_degrees]
    # We assume all inner hexagons are aligned with angle 0
    inner = [[x, y, 0.0] for x, y in inner_centers]
    
    # Calculate minimal outer side length S
    # Outer hexagon centered at (0,0), angle 0.
    # Normals to edges of outer hexagon are at angles 30, 90, 150, 210, 270, 330 degrees.
    # In radians: pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
    
    normals = []
    for k in range(6):
        angle = math.pi / 6 + k * math.pi / 3
        normals.append((math.cos(angle), math.sin(angle)))
        
    max_projection = 0.0
    for nx, ny in normals:
        max_proj_for_normal = 0.0
        for x, y in inner_centers:
            proj = x * nx + y * ny
            if proj > max_proj_for_normal:
                max_proj_for_normal = proj
        if max_proj_for_normal > max_projection:
            max_projection = max_proj_for_normal
            
    # Condition: max_proj + apothem_inner <= S * apothem_outer
    # apothem_inner = sqrt(3)/2
    # apothem_outer = S * sqrt(3)/2
    # max_proj + sqrt(3)/2 <= S * sqrt(3)/2
    # S >= (max_proj + sqrt(3)/2) / (sqrt(3)/2)
    # S >= max_proj * 2/sqrt(3) + 1
    
    S = max_projection * (2.0 / math.sqrt(3)) + 1.0
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
