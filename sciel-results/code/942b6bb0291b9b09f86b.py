import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define 11 positions in axial coordinates (q, r) for a compact hexagonal cluster
    # Center (0,0)
    # Ring 1: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Ring 2 (partial): (2,0), (-2,0), (0,2), (0,-2) -- wait, this makes it a cross.
    # Let's use the 3-4-4 row structure or similar compact one.
    # Axial coords for 11 points:
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (0,-1)
    # (1,-1), (-1,1)
    # (2,0), (-2,0)
    # (0,2), (0,-2)
    # This is 11 points.
    # Let's check if they are disjoint.
    # Distance between (0,0) and (1,0) is sqrt(3).
    # Distance between (1,0) and (2,0) is sqrt(3).
    # Distance between (1,0) and (0,1) is sqrt(3).
    # Distance between (1,0) and (1,-1) is sqrt(3).
    # Distance between (0,1) and (0,2) is sqrt(3).
    # This forms a valid packing.
    
    axial_points = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    # Gap factor to ensure strict disjointness
    gap = 1.001
    sqrt3 = math.sqrt(3)
    
    centers = []
    for q, r in axial_points:
        x = sqrt3 * gap * (q + r / 2.0)
        y = 1.5 * gap * r
        centers.append((x, y))
        
    # Calculate the required outer side length S
    # Outer hexagon is centered at (0,0) with angle 0.
    # Inner hexagons are at centers with angle 0.
    # Vertices of a unit hexagon at (cx, cy) with angle 0:
    # (cx+1, cy), (cx+0.5, cy+sqrt(3)/2), (cx-0.5, cy+sqrt(3)/2),
    # (cx-1, cy), (cx-0.5, cy-sqrt(3)/2), (cx+0.5, cy-sqrt(3)/2)
    
    max_S = 0.0
    
    for cx, cy in centers:
        # Generate 6 vertices
        verts = [
            (cx + 1.0, cy),
            (cx + 0.5, cy + sqrt3 / 2.0),
            (cx - 0.5, cy + sqrt3 / 2.0),
            (cx - 1.0, cy),
            (cx - 0.5, cy - sqrt3 / 2.0),
            (cx + 0.5, cy - sqrt3 / 2.0)
        ]
        
        for vx, vy in verts:
            # Calculate required S for this point (vx, vy)
            # Constraints:
            # 1. S >= vx + vy/sqrt3
            # 2. S >= -vx + vy/sqrt3
            # 3. S >= -vx - vy/sqrt3
            # 4. S >= vx - vy/sqrt3
            # 5. S >= 2*vy/sqrt3
            # 6. S >= -2*vy/sqrt3
            
            s1 = vx + vy / sqrt3
            s2 = -vx + vy / sqrt3
            s3 = -vx - vy / sqrt3
            s4 = vx - vy / sqrt3
            s5 = 2 * vy / sqrt3
            s6 = -2 * vy / sqrt3
            
            local_max = max(s1, s2, s3, s4, s5, s6)
            if local_max > max_S:
                max_S = local_max
                
    # Add a tiny epsilon to S to ensure containment due to floating point
    outer_side = max_S + 1e-9
    
    # Format inner list: [[x, y, angle], ...]
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
