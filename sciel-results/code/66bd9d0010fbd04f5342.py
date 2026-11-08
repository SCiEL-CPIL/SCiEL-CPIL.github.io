import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # 1. Generate inner hexagons
    # 3 rows, 4 columns
    # Flat-top orientation (angle 30 deg relative to pointy-right 0 deg)
    # Lattice spacing: dx = sqrt(3), dy = 1.5
    inner = []
    for r in range(3):
        for q in range(4):
            # Axial coords to Cartesian for flat-top lattice
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            inner.append([x, y, 30.0])
            
    # 2. Get all vertices of inner hexagons
    points = []
    for cx, cy, angle_deg in inner:
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            v_angle = angle_rad + math.radians(60 * i)
            vx = cx + math.cos(v_angle)
            vy = cy + math.sin(v_angle)
            points.append((vx, vy))
            
    points = np.array(points)
    
    # 3. Find minimal enclosing regular hexagon
    best_S = float('inf')
    best_center = (0,0)
    best_angle = 0.0
    
    # Centroid of points
    centroid = points.mean(axis=0)
    
    # Try different rotations of the outer hexagon
    # 0 to 60 degrees in steps of 5
    for a_deg in range(0, 61, 5):
        a_rad = math.radians(a_deg)
        # Normals for the 6 sides of the outer hexagon
        # Side k is perpendicular to angle a_rad + k*60
        normals = []
        for k in range(6):
            n_angle = a_rad + math.radians(60 * k)
            normals.append((math.cos(n_angle), math.sin(n_angle)))
        
        # For a fixed center C, the required inradius R_in is:
        # R_in >= max_k ( max_p ((p - C) . n_k) )
        # S = R_in * 2 / sqrt(3)
        
        # We estimate C as the centroid. 
        # To be safer, we can slightly adjust C or just use centroid.
        # Given the symmetry, centroid is a very good approximation.
        
        C = centroid
        max_proj = 0.0
        for n in normals:
            # Project all points onto normal n, subtract C
            # proj = (p - C) . n
            # max_proj_k = max(proj)
            # We need R_in >= max_proj_k
            projs = points.dot(np.array(n)) - C.dot(np.array(n))
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
        # If max_proj is negative (all points behind center relative to normal), 
        # it means the hexagon is huge or C is outside. But C is centroid, so this shouldn't happen for enclosing.
        # Actually, if C is centroid, projections will be positive for some normals.
        
        S = max_proj * 2.0 / sqrt3
        
        if S < best_S:
            best_S = S
            best_center = C
            best_angle = a_deg
            
    # Add a small margin for safety
    best_S += 0.01
    
    return inner, (best_center[0], best_center[1]), best_S, best_angle
# EVOLVE-BLOCK-END
