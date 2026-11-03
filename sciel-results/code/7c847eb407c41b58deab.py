import math

def construct():
    sqrt3 = math.sqrt(3)
    
    # Core 7
    inner = [
        [0.0, 0.0, 0.0],
        [sqrt3, 0.0, 0.0],
        [sqrt3/2, 1.5, 0.0],
        [-sqrt3/2, 1.5, 0.0],
        [-sqrt3, 0.0, 0.0],
        [-sqrt3/2, -1.5, 0.0],
        [sqrt3/2, -1.5, 0.0]
    ]
    
    # 4 Outward (Symmetric placement for 11 total? 
    # The previous code was asymmetric. Let's try to make it more compact/symmetric or just fix the gap.
    # To ensure we have 11 hexagons, we keep the 4 outward ones.
    # To improve symmetry, we might shift them, but let's first fix the overlap.
    
    # Apply GAP factor to ensure strict disjointness
    gap_factor = 1.001
    inner_scaled = []
    for x, y, a in inner:
        inner_scaled.append([x * gap_factor, y * gap_factor, a])
    
    # Add the 4 outward hexagons, also scaled
    # Previous positions:
    # [2*sqrt3, 0.0, 0.0]
    # [sqrt3, 3.0, 0.0]
    # [-sqrt3, 3.0, 0.0]
    # [-2*sqrt3, 0.0, 0.0]
    
    outward = [
        [2*sqrt3, 0.0, 0.0],
        [sqrt3, 3.0, 0.0],
        [-sqrt3, 3.0, 0.0],
        [-2*sqrt3, 0.0, 0.0]
    ]
    
    for x, y, a in outward:
        inner_scaled.append([x * gap_factor, y * gap_factor, a])
        
    inner = inner_scaled
    
    # Function to check if a point (vx, vy) is inside a regular hexagon
    # centered at (cx, cy) with side length s and rotation angle theta (radians)
    def point_in_hex(px, py, cx, cy, s, theta):
        # Translate point to hexagon center
        dx = px - cx
        dy = py - cy
        # Rotate point by -theta to align hexagon with axes
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        
        # Hexagon vertices for side s, rotation 0:
        # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # Actually, easier to check distance to edges.
        # The hexagon is the intersection of 6 half-planes.
        # For a hexagon centered at origin, flat-top (rotation 0):
        # Inradius r = s * sqrt(3) / 2
        # The boundaries are:
        # x <= s
        # x >= -s
        # sqrt(3)*x + y <= s*sqrt(3)  (Wait, vertices are at (s,0)... no, flat top has vertices at (s/2, r) etc? No.)
        
        # Let's use the standard "flat top" hexagon with side s.
        # Vertices: (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # This is "pointy top" in some conventions, but let's stick to coordinates.
        # Inradius (distance from center to edge midpoint) = s * sqrt(3) / 2.
        # The 6 edges are at distance r from center.
        # Edge 1 (right): x = s. Normal (1, 0). Check: rx <= s.
        # Edge 2 (top-right): line through (s,0) and (s/2, s*sqrt(3)/2).
        #   Slope: (s*sqrt(3)/2 - 0) / (s/2 - s) = (s*sqrt(3)/2) / (-s/2) = -sqrt(3).
        #   Eq: y - 0 = -sqrt(3)(x - s) => sqrt(3)x + y = s*sqrt(3).
        #   Check: sqrt(3)rx + ry <= s*sqrt(3).
        # Edge 3 (top-left): line through (s/2, s*sqrt(3)/2) and (-s/2, s*sqrt(3)/2).
        #   Eq: y = s*sqrt(3)/2.
        #   Check: ry <= s*sqrt(3)/2.
        # Edge 4 (left): x = -s. Check: rx >= -s.
        # Edge 5 (bottom-left): line through (-s,0) and (-s/2, -s*sqrt(3)/2).
        #   Slope: (-s*sqrt(3)/2 - 0) / (-s/2 - (-s)) = (-s*sqrt(3)/2) / (s/2) = -sqrt(3).
        #   Eq: y - 0 = -sqrt(3)(x - (-s)) => y = -sqrt(3)(x + s) => sqrt(3)x + y = -s*sqrt(3).
        #   Check: sqrt(3)rx + ry >= -s*sqrt(3).
        # Edge 6 (bottom-right): line through (-s/2, -s*sqrt(3)/2) and (s/2, -s*sqrt(3)/2).
        #   Eq: y = -s*sqrt(3)/2.
        #   Check: ry >= -s*sqrt(3)/2.
        
        r = s * sqrt3 / 2
        if rx > s: return False
        if rx < -s: return False
        if ry > r: return False
        if ry < -r: return False
        if sqrt3 * rx + ry > s * sqrt3: return False
        if sqrt3 * rx + ry < -s * sqrt3: return False
        
        return True

    # To find the minimum outer side length, we need to find the smallest s such that
    # all inner hexagons are inside the outer hexagon.
    # This is a non-trivial optimization.
    # Approximation: The outer hexagon must contain the bounding box of the inner hexagons.
    # A simpler approach: The distance from the outer center to the farthest vertex of any inner hexagon
    # must be less than or equal to the circumradius of the outer hexagon.
    # Circumradius of outer hexagon = outer_side.
    # However, the outer hexagon can be rotated.
    
    # Let's compute the maximum distance from (0,0) to any vertex of any inner hexagon.
    # Then set outer_side = max_dist + small_epsilon.
    # But this assumes the outer hexagon is a circle.
    # For a hexagon, the "radius" varies with angle.
    # Min circumradius for a given angle?
    
    # Let's iterate over outer rotations theta_outer in [0, pi/3].
    # For each theta_outer, find the minimum s such that all inner hexagons are inside.
    # This can be done by checking the constraints.
    # Constraint: For all inner hexagons i, and all vertices v of i, point_in_hex(v, 0, 0, s, theta_outer) is True.
    # Actually, it's sufficient to check the vertices of the inner hexagons.
    
    min_side = float('inf')
    best_theta = 0.0
    
    # Pre-compute vertices of unit hexagons (side 1, rotation 0)
    # Vertices: (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
    unit_vertices = [
        (1.0, 0.0),
        (0.5, sqrt3/2),
        (-0.5, sqrt3/2),
        (-1.0, 0.0),
        (-0.5, -sqrt3/2),
        (0.5, -sqrt3/2)
    ]
    
    # For each inner hexagon, pre-compute its vertices in global coords
    inner_vertices = []
    for x, y, a_deg in inner:
        a_rad = math.radians(a_deg)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        verts = []
        for vx, vy in unit_vertices:
            # Rotate vertex by a_rad
            rx = vx * cos_a - vy * sin_a
            ry = vx * sin_a + vy * cos_a
            # Translate
            gx = x + rx
            gy = y + ry
            verts.append((gx, gy))
        inner_vertices.append(verts)
        
    # Search over outer rotation
    num_steps = 60
    for i in range(num_steps + 1):
        theta_outer = (i / num_steps) * (math.pi / 3)
        
        # Find minimum s
        # s must be such that all vertices are inside.
        # The condition point_in_hex(v, 0, 0, s, theta) depends on s linearly in the inequalities.
        # rx <= s
        # rx >= -s
        # ry <= s*sqrt3/2
        # ry >= -s*sqrt3/2
        # sqrt3*rx + ry <= s*sqrt3
        # sqrt3*rx + ry >= -s*sqrt3
        
        # So s >= max( |rx|, |ry|*2/sqrt3, |sqrt3*rx + ry|/sqrt3 ) for all vertices.
        
        max_s = 0.0
        for verts in inner_vertices:
            for vx, vy in verts:
                # Transform to outer hexagon frame
                dx = vx
                dy = vy
                cos_t = math.cos(-theta_outer)
                sin_t = math.sin(-theta_outer)
                rx = dx * cos_t - dy * sin_t
                ry = dx * sin_t + dy * cos_t
                
                s1 = abs(rx)
                s2 = abs(ry) * 2.0 / sqrt3
                s3 = abs(sqrt3 * rx + ry) / sqrt3
                
                max_s = max(max_s, s1, s2, s3)
                
        if max_s < min_side:
            min_side = max_s
            best_theta = theta_outer
            
    # Add a small safety margin
    min_side *= 1.001
    
    return inner, (0.0, 0.0), min_side, math.degrees(best_theta)

