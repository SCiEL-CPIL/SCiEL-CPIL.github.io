import math

def construct():
    # Structural Change: Use a "brick-wall" rectangular grid layout
    # instead of the triangular lattice.
    # Flat-top hexagons, side 1.
    # Horizontal spacing: 2.0 (touching at vertices)
    # Vertical spacing: sqrt(3) (touching at sides)
    # Offset odd rows by 1.0 (half of horizontal spacing)
    
    sqrt3 = math.sqrt(3.0)
    
    centers = []
    
    # Row 0: y = 0, x = 0, 2, 4, 6
    for x in [0.0, 2.0, 4.0, 6.0]:
        centers.append((x, 0.0))
        
    # Row 1: y = sqrt(3), x = 1, 3, 5 (offset by 1)
    for x in [1.0, 3.0, 5.0]:
        centers.append((x, sqrt3))
        
    # Row 2: y = 2*sqrt(3), x = 0, 2, 4, 6
    for x in [0.0, 2.0, 4.0, 6.0]:
        centers.append((x, 2.0 * sqrt3))
        
    # Center the cluster at (0,0)
    # X range: 0 to 6 -> center 3
    # Y range: 0 to 2*sqrt(3) -> center sqrt(3)
    cx_offset = 3.0
    cy_offset = sqrt3
    
    inner = []
    all_vertices = []
    
    h_half = sqrt3 / 2.0
    
    for cx, cy in centers:
        cx = cx - cx_offset
        cy = cy - cy_offset
        inner.append([cx, cy, 0.0])
        
        # Generate 6 vertices of the unit flat-top hexagon
        # (1, 0), (0.5, h_half), (-0.5, h_half), (-1, 0), (-0.5, -h_half), (0.5, -h_half)
        verts = [
            (1.0, 0.0),
            (0.5, h_half),
            (-0.5, h_half),
            (-1.0, 0.0),
            (-0.5, -h_half),
            (0.5, -h_half)
        ]
        
        for vx, vy in verts:
            all_vertices.append((cx + vx, cy + vy))
            
    # Calculate minimum side length S for a flat-top regular hexagon
    # centered at (0,0) to contain all vertices.
    # Inequalities for point (x,y) to be inside flat-top hex of side S:
    # 1. |y| <= S * sqrt(3) / 2  => S >= 2 * |y| / sqrt(3)
    # 2. |x + y/sqrt(3)| <= S    => S >= |x + y/sqrt(3)|
    # 3. |x - y/sqrt(3)| <= S    => S >= |x - y/sqrt(3)|
    
    max_S = 0.0
    inv_sqrt3 = 1.0 / sqrt3
    
    for x, y in all_vertices:
        s1 = 2.0 * abs(y) * inv_sqrt3
        s2 = abs(x + y * inv_sqrt3)
        s3 = abs(x - y * inv_sqrt3)
        
        local_max = max(s1, s2, s3)
        if local_max > max_S:
            max_S = local_max
            
    # Add small epsilon for strict containment
    side = max_S + 0.001
    
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
