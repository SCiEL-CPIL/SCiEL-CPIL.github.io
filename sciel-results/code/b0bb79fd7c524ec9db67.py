import numpy as np

def construct():
    # 11 unit hexagons
    # 1 central, 6 ring, 4 bay
    
    inner = []
    
    # Central
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1 (6 hexes)
    r1 = np.sqrt(3)
    for i in range(6):
        angle = i * 60.0
        rad = np.deg2rad(angle)
        x = r1 * np.cos(rad)
        y = r1 * np.sin(rad)
        inner.append([x, y, 0.0])
        
    # Ring 2 (4 hexes in bays)
    r2 = 3.0
    bay_angles = [30.0, 90.0, 150.0, 210.0]
    for angle in bay_angles:
        rad = np.deg2rad(angle)
        x = r2 * np.cos(rad)
        y = r2 * np.sin(rad)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length approx 4.002, use 4.1 for safety
    side = 4.1
    center = (0.0, 0.0)
    # Rotate outer hexagon by 30 degrees to align vertices with spikes
    angle = 30.0
    
    return inner, center, side, angle
