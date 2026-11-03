import numpy as np

N = 12

def construct():
    # Create a 4x3 block of hexagons
    # Axial coordinates (q, r)
    # 4 rows (r=0..3), 3 cols (q=0..2)
    coords = []
    for r in range(4):
        for q in range(3):
            # Convert axial to cartesian
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            coords.append([x, y])
    
    coords = np.array(coords)
    
    # Center the block
    center = coords.mean(axis=0)
    coords = coords - center
    
    # Rotate by 15 degrees to better fit in the hexagon
    angle = np.deg2rad(15.0)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rotated = []
    for x, y in coords:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        rotated.append([rx, ry, 0.0]) # angle_degrees 0.0 for inner hexagons
        
    inner = rotated
    outer_center = (0.0, 0.0)
    outer_side = 4.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
