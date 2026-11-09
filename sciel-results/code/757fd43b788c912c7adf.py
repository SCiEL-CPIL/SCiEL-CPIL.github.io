import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Axial coordinates for 11 hexagons
    # Center + Ring 1 + 4 corners
    axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    
    inner = []
    for q, r in axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Calculate minimal side length S
    # Outer hexagon centered at (0,0), rotation 0 (pointy top)
    # Normals at 0, 60, 120, 180, 240, 300 degrees
    normals = []
    for i in range(6):
        theta = np.deg2rad(i * 60)
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for hx, hy, ha in inner:
        # Vertices of inner hexagon (pointy top, rotation 0)
        # Angles: 30, 90, 150, 210, 270, 330
        for j in range(6):
            v_theta = np.deg2rad(30 + j * 60)
            vx = hx + np.cos(v_theta)
            vy = hy + np.sin(v_theta)
            v = np.array([vx, vy])
            projs = v @ normals
            max_proj = max(max_proj, np.max(projs))
            
    # A = max_proj
    # S * sqrt(3) / 2 = A
    S = max_proj * 2.0 / np.sqrt(3)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
