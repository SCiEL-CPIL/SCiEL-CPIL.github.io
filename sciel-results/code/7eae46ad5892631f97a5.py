"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # 10s is plenty, limit is 120s
    
    # Generate lattice points
    points = []
    max_coord = 5
    sqrt3 = np.sqrt(3)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = sqrt3 * (i + j / 2.0)
            y = 1.5 * j
            dist = np.sqrt(x*x + y*y)
            points.append((dist, i, j, x, y))
            
    points.sort(key=lambda p: p[0])
    selected = points[:N]
    
    # Container normals
    normals = []
    for k in range(6):
        angle = np.deg2rad(30 + 60 * k)
        normals.append((np.cos(angle), np.sin(angle)))
        
    # Inner hexagon vertices
    vertices = []
    for j in range(6):
        angle = np.deg2rad(60 * j)
        vertices.append((np.cos(angle), np.sin(angle)))
        
    max_s = 0.0
    inner_coords = []
    
    for dist, i, j, x, y in selected:
        cx = x
        cy = y
        for vx, vy in vertices:
            px = cx + vx
            py = cy + vy
            for nx, ny in normals:
                proj = abs(px * nx + py * ny)
                s_needed = 2.0 * proj / sqrt3
                if s_needed > max_s:
                    max_s = s_needed
        inner_coords.append([x, y, 0.0])
        
    final_s = max_s + 0.001
    return inner_coords, (0.0, 0.0), final_s, 0.0
# EVOLVE-BLOCK-END
