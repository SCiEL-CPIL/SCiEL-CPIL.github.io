import numpy as np

N = 12

def construct():
    # Core 7 hexagons
    centers = [
        (0.0, 0.0),
        (2.0, 0.0),
        (1.0, np.sqrt(3)),
        (-1.0, np.sqrt(3)),
        (-2.0, 0.0),
        (-1.0, -np.sqrt(3)),
        (1.0, -np.sqrt(3))
    ]
    
    # 5 Corner hexagons from Layer 2
    # Corners are at distance 4 from origin
    # Angles: 0, 60, 120, 180, 240, 300
    # We pick 5 of them
    corner_angles = [0, 60, 120, 180, 240]
    for deg in corner_angles:
        rad = np.deg2rad(deg)
        x = 4.0 * np.cos(rad)
        y = 4.0 * np.sin(rad)
        centers.append((x, y))
        
    # Total 12 centers
    # All angles 0
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    # Side length 5.0 is tight for this configuration
    # Add small epsilon to avoid boundary issues
    side = 5.01
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
