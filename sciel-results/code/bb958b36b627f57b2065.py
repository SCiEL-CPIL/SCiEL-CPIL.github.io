import numpy as np

N = 11

def construct():
    # Place 11 unit hexagons on a hex grid in a compact 3-5-3 arrangement
    hexes = [
        (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1)
    ]
    
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in hexes:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for cx, cy in centers:
        for angle_deg in range(0, 360, 60):
            angle = np.radians(angle_deg)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # Minimum enclosing regular hexagon: center at origin, 0 rotation
    # Edge normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [np.radians(a) for a in [30, 90, 150, 210, 270, 330]]
    
    max_val = 0.0
    for vx, vy in all_vertices:
        for na in normal_angles:
            val = vx * np.cos(na) + vy * np.sin(na)
            if val > max_val:
                max_val = val
    
    R = max_val / (sqrt3 / 2.0)
    
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    return inner, (0.0, 0.0), R, 0.0
