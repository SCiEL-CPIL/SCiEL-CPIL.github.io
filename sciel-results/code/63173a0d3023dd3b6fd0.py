import math

def construct():
    N = 11
    
    # Determine the size of the hexagonal grid needed to hold N hexagons
    # N_hex(k) = 3*k*(k-1) + 1
    k = 1
    while 3 * k * (k - 1) + 1 < N:
        k += 1
        
    # Generate axial coordinates for a hexagon of side k
    # Condition: max(|q|, |r|, |q+r|) <= k-1
    centers_axial = []
    for q in range(-(k - 1), k):
        for r in range(-(k - 1), k):
            if max(abs(q), abs(r), abs(q + r)) <= k - 1:
                centers_axial.append((q, r))
                
    # Convert to Cartesian and sort by distance from origin to keep the cluster compact
    centers_cart = []
    for q, r in centers_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        dist = x * x + y * y
        centers_cart.append((dist, x, y))
        
    # Sort by distance
    centers_cart.sort(key=lambda item: item[0])
    
    # Select the first N centers
    selected_centers = [(x, y) for _, x, y in centers_cart[:N]]
    
    # Calculate the required outer side length S
    # The outer hexagon is flat-top, centered at (0,0), aligned with inner hexagons.
    # For a point (x, y) to be inside a flat-top hexagon of side S, it must satisfy:
    # 1. x + y/sqrt(3) <= S
    # 2. -x + y/sqrt(3) <= S
    # 3. 2*y/sqrt(3) <= S
    # 4. -x - y/sqrt(3) <= S
    # 5. x - y/sqrt(3) <= S
    # 6. -2*y/sqrt(3) <= S
    
    sqrt3 = math.sqrt(3)
    max_s = 0.0
    
    for cx, cy in selected_centers:
        # Vertices of a unit hexagon centered at (cx, cy)
        # (1, 0), (-1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (0.5, -sqrt(3)/2), (-0.5, -sqrt(3)/2)
        vertices = [
            (cx + 1.0, cy),
            (cx - 1.0, cy),
            (cx + 0.5, cy + sqrt3 / 2.0),
            (cx - 0.5, cy + sqrt3 / 2.0),
            (cx + 0.5, cy - sqrt3 / 2.0),
            (cx - 0.5, cy - sqrt3 / 2.0)
        ]
        
        for vx, vy in vertices:
            # Calculate required S for this vertex
            s1 = vx + vy / sqrt3
            s2 = -vx + vy / sqrt3
            s3 = 2.0 * vy / sqrt3
            s4 = -vx - vy / sqrt3
            s5 = vx - vy / sqrt3
            s6 = -2.0 * vy / sqrt3
            
            s_req = max(s1, s2, s3, s4, s5, s6)
            if s_req > max_s:
                max_s = s_req
                
    # The outer hexagon side length is max_s
    side_length = max_s
    
    # Construct the inner hexagon list
    # Format: [x, y, angle_degrees]
    inner = []
    for cx, cy in selected_centers:
        inner.append([cx, cy, 0.0])
        
    # Outer hexagon parameters
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, side_length, outer_angle
