import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    
    sqrt3 = np.sqrt(3)
    
    # Generate lattice points
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = sqrt3 * (i + 0.5 * j)
            y = 1.5 * j
            dist_sq = x*x + y*y
            points.append((dist_sq, i, j, x, y))
            
    points.sort(key=lambda p: p[0])
    selected = points[:N]
    
    inner_hexes = [[x, y, 0.0] for _, i, j, x, y in selected]
    
    # Pre-compute vertices for each inner hexagon
    # Vertices are relative to center
    rel_vertices = []
    for k in range(6):
        theta = np.deg2rad(60 * k)
        rel_vertices.append((np.cos(theta), np.sin(theta)))
    rel_vertices = np.array(rel_vertices)
    
    best_s = float('inf')
    best_angle = 0.0
    
    # Try different outer hexagon orientations
    for angle_deg in [0, 15, 30, 45, 60]:
        alpha = np.deg2rad(angle_deg)
        # Rotation matrix to rotate points by -alpha
        # x' = x cos(alpha) + y sin(alpha)
        # y' = -x sin(alpha) + y cos(alpha)
        # This aligns the outer hexagon with the standard one
        
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        
        max_s = 0.0
        for x, y, _ in inner_hexes:
            # Rotate all vertices of this hexagon
            # Vertex = (x + vx, y + vy)
            # Rotated Vertex = ( (x+vx)cos + (y+vy)sin, -(x+vx)sin + (y+vy)cos )
            # We can just rotate the center and the relative vertices
            cx = x * cos_a + y * sin_a
            cy = -x * sin_a + y * cos_a
            
            # Relative vertices rotated
            # vx' = vx cos + vy sin
            # vy' = -vx sin + vy cos
            rot_rel_vx = rel_vertices[:, 0] * cos_a + rel_vertices[:, 1] * sin_a
            rot_rel_vy = -rel_vertices[:, 0] * sin_a + rel_vertices[:, 1] * cos_a
            
            vx_all = cx + rot_rel_vx
            vy_all = cy + rot_rel_vy
            
            # Check constraints for standard hexagon (angle 0)
            # 1. |vy| <= S * sqrt(3) / 2
            s1 = 2.0 * np.max(np.abs(vy_all)) / sqrt3
            
            # 2. |sqrt(3)/2 * vx + 1/2 * vy| <= S * sqrt(3) / 2
            val2 = np.abs(sqrt3/2.0 * vx_all + 0.5 * vy_all)
            s2 = 2.0 * np.max(val2) / sqrt3
            
            # 3. |sqrt(3)/2 * vx - 1/2 * vy| <= S * sqrt(3) / 2
            val3 = np.abs(sqrt3/2.0 * vx_all - 0.5 * vy_all)
            s3 = 2.0 * np.max(val3) / sqrt3
            
            s_req = max(s1, s2, s3)
            if s_req > max_s:
                max_s = s_req
                
        if max_s < best_s:
            best_s = max_s
            best_angle = angle_deg

    best_s += 1e-6
    
    return inner_hexes, (0.0, 0.0), best_s, best_angle
