import math
import time

def construct():
    start_time = time.time()
    
    sqrt3 = math.sqrt(3.0)
    
    # Define 11 specific centers for a compact packing
    # Axial coordinates (q, r)
    axial_centers = [
        (0, 0),   # Center
        (0, 1),   # N
        (1, 0),   # NE
        (1, -1),  # SE
        (0, -1),  # S
        (-1, 0),  # SW
        (-1, 1),  # NW
        (1, 1),   # Add 1
        (-1, -1), # Add 2
        (2, -1),  # Add 3
        (-2, 1)   # Add 4
    ]
    
    centers = []
    for q, r in axial_centers:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
        
    inner = []
    all_vertices = []
    
    h_half = sqrt3 / 2.0
    
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
        # Generate 6 vertices of the unit flat-top hexagon
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
    
    # Safety check for time
    if time.time() - start_time > 110:
         return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.001, 0.0

    return inner, center, side, outer_angle
