import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    inner = []
    
    # 1. Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. Layer 1: 6 hexagons at distance sqrt(3)
    dist_l1 = np.sqrt(3)
    for k in range(6):
        angle_rad = k * np.pi / 3.0
        x = dist_l1 * np.cos(angle_rad)
        y = dist_l1 * np.sin(angle_rad)
        angle_deg = np.degrees(angle_rad)
        inner.append([x, y, angle_deg])
        
    # 3. Layer 2 (Edge-Mids): 5 hexagons at distance 3
    # Angles: 30, 90, 150, 210, 270 (dropping 330)
    dist_l2 = 3.0
    angles_deg = [30, 90, 150, 210, 270]
    for angle_deg in angles_deg:
        angle_rad = np.radians(angle_deg)
        x = dist_l2 * np.cos(angle_rad)
        y = dist_l2 * np.sin(angle_rad)
        inner.append([x, y, angle_deg])
        
    # Outer hexagon
    # Side length S = 3 + sqrt(3)/2
    side = 3.0 + np.sqrt(3) / 2.0
    center = (0.0, 0.0)
    # Rotation 0.0 assumes normals are at 0, 60, ... which puts vertices at 30, 90, ...
    # This aligns outer vertices with the farthest inner points (Edge-Mids)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
