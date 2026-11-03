import numpy as np

N = 11

def construct():
    sqrt3 = np.sqrt(3)
    
    # Flat-top hexagonal lattice basis vectors
    u = np.array([1.5, sqrt3 / 2])
    v = np.array([0.0, sqrt3])
    
    # 11 hexagon positions in axial coordinates (compact cluster)
    axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (1, 1), (-2, 2), (-2, 1)]
    
    centers = [q * u + r * v for q, r in axial]
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    
    # Generate all vertices of inner hexagons (flat-top, side=1)
    all_verts = []
    for cx, cy, _ in inner:
        for i in range(6):
            angle = np.radians(60 * i)
            all_verts.append((cx + np.cos(angle), cy + np.sin(angle)))
    all_verts = np.array(all_verts)
    
    # Find minimum outer hexagon side by trying rotations
    best_S = float('inf')
    best_rot = 0.0
    
    for rot_deg in range(0, 360, 3):
        rot = np.radians(rot_deg)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        
        x_rot = all_verts[:, 0] * cos_r + all_verts[:, 1] * sin_r
        y_rot = -all_verts[:, 0] * sin_r + all_verts[:, 1] * cos_r
        
        bounds = np.maximum.reduce([
            np.abs(x_rot),
            x_rot + y_rot / sqrt3,
            -x_rot + y_rot / sqrt3,
            -x_rot - y_rot / sqrt3,
            x_rot - y_rot / sqrt3
        ])
        
        S = float(np.max(bounds))
        if S < best_S:
            best_S = S
            best_rot = float(rot_deg)
    
    return inner, (0.0, 0.0), best_S, best_rot
