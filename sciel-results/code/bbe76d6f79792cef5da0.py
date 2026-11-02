import numpy as np

N = 11

def construct():
    # Core 7 hexagons
    # Center
    centers = [[0.0, 0.0, 0.0]]
    
    # 6 Ring Hexagons
    # Distance sqrt(3), angles 30, 90, 150, 210, 270, 330
    dist = np.sqrt(3)
    angles_ring = [30, 90, 150, 210, 270, 330]
    for ang in angles_ring:
        rad = np.deg2rad(ang)
        x = dist * np.cos(rad)
        y = dist * np.sin(rad)
        centers.append([x, y, 0.0])
        
    # 4 Gap Hexagons
    # Gaps at 0, 60, 120, 180, 240, 300
    # Pick 0, 60, 180, 240
    gap_angles = [0, 60, 180, 240]
    gap_dist = 3.0
    for ang in gap_angles:
        rad = np.deg2rad(ang)
        x = gap_dist * np.cos(rad)
        y = gap_dist * np.sin(rad)
        centers.append([x, y, 0.0])
        
    # Outer Hexagon
    # Side length approx 4.2
    # Rotation 30 degrees to align corners with gap directions
    outer_center = (0.0, 0.0)
    outer_side = 4.2
    outer_angle = 30.0
    
    return centers, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-START
def construct():
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
