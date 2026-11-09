import numpy as np

def construct():
    # Define centers of 11 unit hexagons
    # 7 core + 4 extension
    centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (-np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (np.sqrt(3)/2, -1.5),
        (-np.sqrt(3)/2, -1.5),
        (2*np.sqrt(3), 0.0),
        (-2*np.sqrt(3), 0.0),
        (1.5*np.sqrt(3), -1.5),
        (-1.5*np.sqrt(3), -1.5)
    ]
    
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # Calculate minimal side length S
    # Outer hexagon is flat-top (angle 0)
    # Face normals are at 30, 90, 150, 210, 270, 330 degrees
    normals = [
        (np.cos(np.radians(30)), np.sin(np.radians(30))),
        (np.cos(np.radians(90)), np.sin(np.radians(90))),
        (np.cos(np.radians(150)), np.sin(np.radians(150))),
        (np.cos(np.radians(210)), np.sin(np.radians(210))),
        (np.cos(np.radians(270)), np.sin(np.radians(270))),
        (np.cos(np.radians(330)), np.sin(np.radians(330)))
    ]
    
    max_proj = 0.0
    for cx, cy, ang in inner:
        # Vertices of this inner hexagon
        for k in range(6):
            theta = np.radians(ang + 60 * k)
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            for nx, ny in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
                    
    # S * sqrt(3)/2 >= max_proj
    S = (2.0 / np.sqrt(3)) * max_proj
    
    return inner, (0.0, 0.0), S, 0.0
