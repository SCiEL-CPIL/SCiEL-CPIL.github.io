import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    sqrt3 = math.sqrt(3)
    
    # Centers for 11 hexagons in a cross pattern
    centers = [
        (0.0, 0.0),
        (1.5, sqrt3/2),
        (-1.5, sqrt3/2),
        (0.0, sqrt3),
        (1.5, -sqrt3/2),
        (-1.5, -sqrt3/2),
        (0.0, -sqrt3),
        (3.0, 0.0),
        (-3.0, 0.0),
        (0.0, 2*sqrt3),
        (0.0, -2*sqrt3)
    ]
    
    # Local vertices of unit hexagon
    inner_vertices_local = [
        (1.0, 0.0),
        (0.5, sqrt3/2),
        (-0.5, sqrt3/2),
        (-1.0, 0.0),
        (-0.5, -sqrt3/2),
        (0.5, -sqrt3/2)
    ]
    
    best_L = float('inf')
    best_angle = 0.0
    
    # Try a few outer hexagon angles
    for angle_deg in [0, 15, 30, 45, 60]:
        angle_rad = math.radians(angle_deg)
        # Normals for outer hexagon
        # Vertices at angle_deg + k*60.
        # Sides perpendicular to vertices? No.
        # Side midpoints are at angle_deg + 30 + k*60.
        # Normals point to side midpoints.
        normal_angles = [angle_rad + math.radians(30 + 60*k) for k in range(6)]
        normals = [(math.cos(a), math.sin(a)) for a in normal_angles]
        
        max_proj = 0.0
        
        for cx, cy in centers:
            for vx, vy in inner_vertices_local:
                px = cx + vx
                py = cy + vy
                for nx, ny in normals:
                    proj = abs(px*nx + py*ny)
                    if proj > max_proj:
                        max_proj = proj
                        
        L = max_proj * 2.0 / sqrt3
        
        if L < best_L:
            best_L = L
            best_angle = angle_deg
            
        # Deadline guard
        if time.time() - start_time > 10:
            break
            
    # Prepare output
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    center = (0.0, 0.0)
    
    return inner, center, best_L, best_angle
# EVOLVE-BLOCK-END
