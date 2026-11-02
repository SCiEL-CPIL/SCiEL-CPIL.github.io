import numpy as np
import time

N = 11
SEED_VALUE = 40.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define candidate configurations (axial coordinates)
    configs = []
    
    # Config 1: Star shape
    star = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
        (2,0), (-2,0), (0,2), (0,-2)
    ]
    configs.append(star)
    
    # Config 2: 3x3 block + 2
    block3x3 = [(q,r) for q in range(-1,2) for r in range(-1,2)]
    c2a = block3x3 + [(2,0), (2,-1)]
    configs.append(c2a)
    c2b = block3x3 + [(0,2), (-1,2)]
    configs.append(c2b)
    
    best_score = 0.0
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = SEED_VALUE
    best_angle = 0.0
    
    # Hexagon vertices relative to center (unit hexagon)
    hex_verts_rel = [(np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))) for a in range(0, 360, 60)]
    
    for config in configs:
        if time.time() > deadline:
            break
        
        # Convert to Cartesian
        points = []
        for q, r in config:
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            points.append((x, y))
        
        # Center the cluster
        cx = np.mean([p[0] for p in points])
        cy = np.mean([p[1] for p in points])
        centered_points = [(p[0]-cx, p[1]-cy) for p in points]
        
        # Generate all vertices
        all_vertices = []
        for px, py in centered_points:
            for vx, vy in hex_verts_rel:
                all_vertices.append((px+vx, py+vy))
        
        best_S_for_config = float('inf')
        best_angle_for_config = 0.0
        
        # Try rotations of the outer hexagon
        for angle_deg in np.arange(0, 60, 2.5):
            if time.time() > deadline:
                break
            angle_rad = np.deg2rad(angle_deg)
            normals = []
            for k in range(6):
                a = angle_rad + np.deg2rad(k*60)
                normals.append((np.cos(a), np.sin(a)))
            
            max_proj = 0.0
            for vx, vy in all_vertices:
                for nx, ny in normals:
                    proj = vx*nx + vy*ny
                    if proj > max_proj:
                        max_proj = proj
            
            S_required = (2.0 / np.sqrt(3.0)) * max_proj
            
            if S_required < best_S_for_config:
                best_S_for_config = S_required
                best_angle_for_config = angle_deg
        
        if best_S_for_config < float('inf'):
            score = SEED_VALUE / best_S_for_config
            if score > best_score:
                best_score = score
                best_inner = [[p[0], p[1], 0.0] for p in centered_points]
                best_center = (0.0, 0.0)
                best_side = best_S_for_config
                best_angle = best_angle_for_config

    if best_inner is None:
        # Fallback
        star_pts = []
        for q, r in star:
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            star_pts.append((x, y))
        cx = np.mean([p[0] for p in star_pts])
        cy = np.mean([p[1] for p in star_pts])
        best_inner = [[p[0]-cx, p[1]-cy, 0.0] for p in star_pts]
        best_side = 5.0
        best_angle = 0.0

    return best_inner, best_center, best_side, best_angle
