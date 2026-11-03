import numpy as np
import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    sq3 = math.sqrt(3)
    
    # Lattice vectors for flat-top hexagon tiling
    # These vectors ensure face-to-face contact (no overlap, no gaps)
    va = (1.5, sq3 / 2.0)
    vb = (0.0, sq3)
    
    centers = []
    
    # Center (0,0)
    centers.append((0.0, 0.0))
    
    # 6 Neighbors (Ring 1)
    # Axial coordinates (i, j)
    neighbors = [
        (1, 0), (0, 1), (-1, 1),
        (-1, 0), (0, -1), (1, -1)
    ]
    for i, j in neighbors:
        cx = i * va[0] + j * vb[0]
        cy = i * va[1] + j * vb[1]
        centers.append((cx, cy))
        
    # 4 More (Ring 2)
    # We pick 4 from the next ring to maintain symmetry.
    # (2,0), (0,2), (-2,0), (0,-2) in axial coords
    extra = [
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    for i, j in extra:
        cx = i * va[0] + j * vb[0]
        cy = i * va[1] + j * vb[1]
        centers.append((cx, cy))
        
    # Generate all vertices of the 11 inner hexagons
    # Flat-top unit hexagon vertices relative to center:
    # (1, 0), (0.5, sq3/2), (-0.5, sq3/2), (-1, 0), (-0.5, -sq3/2), (0.5, -sq3/2)
    rel_verts = [
        (1.0, 0.0),
        (0.5, sq3 / 2.0),
        (-0.5, sq3 / 2.0),
        (-1.0, 0.0),
        (-0.5, -sq3 / 2.0),
        (0.5, -sq3 / 2.0)
    ]
    
    all_vertices = []
    for cx, cy in centers:
        for vx, vy in rel_verts:
            all_vertices.append((cx + vx, cy + vy))
            
    # Find minimal S for outer hexagon
    best_S = float('inf')
    best_rotation = 0.0
    
    # The outer hexagon is a regular hexagon.
    # We can rotate it. Due to symmetry, we only need to check 0 to 60 degrees.
    # Step size 1 degree is sufficient for this precision.
    for deg in range(0, 61, 1):
        if time.time() > deadline:
            break
            
        phi = math.radians(deg)
        cos_phi = math.cos(-phi)
        sin_phi = math.sin(-phi)
        
        req_S = 0.0
        for x, y in all_vertices:
            # Rotate point by -phi to align outer hexagon with standard orientation
            xr = x * cos_phi - y * sin_phi
            yr = x * sin_phi + y * cos_phi
            
            # Support function for flat-top hexagon of side S (vertex at S,0)
            # Constraints:
            # 1. |x| <= S
            # 2. |x + y/sqrt(3)| <= S
            # 3. |-x + y/sqrt(3)| <= S
            
            val1 = abs(xr)
            val2 = abs(xr + yr / sq3)
            val3 = abs(-xr + yr / sq3)
            
            # S must be >= max of these values
            local_max = max(val1, val2, val3)
            if local_max > req_S:
                req_S = local_max
                
        if req_S < best_S:
            best_S = req_S
            best_rotation = deg
            
    # Construct return values
    inner = []
    for cx, cy in centers:
        # Inner hexagons are flat-top (0 degrees rotation)
        inner.append([cx, cy, 0.0])
        
    center = (0.0, 0.0)
    side = best_S
    angle = float(best_rotation)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
