import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit hexagon side length is 1.
    # Distance between centers of tangent hexagons is sqrt(3).
    d = np.sqrt(3)
    
    centers = []
    angles = []
    
    # Center
    centers.append((0.0, 0.0))
    angles.append(0.0)
    
    # 6 neighbors
    for i in range(6):
        theta = np.deg2rad(i * 60)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        centers.append((x, y))
        angles.append(0.0)
        
    # 4 more, extending 4 of the lobes
    # Extend lobes at 0, 60, 180, 240 degrees
    ext_indices = [0, 1, 3, 4] # 0, 60, 180, 240
    for i in ext_indices:
        theta = np.deg2rad(i * 60)
        x = 2 * d * np.cos(theta)
        y = 2 * d * np.sin(theta)
        centers.append((x, y))
        angles.append(0.0)
        
    inner = [[x, y, a] for (x, y), a in zip(centers, angles)]
    
    # Calculate required outer side length
    # Max radius is approx 4.464
    # Max projection is approx 3.86
    # S must be at least max(4.464, 3.86 * 2 / sqrt(3))
    # 3.86 * 1.1547 = 4.456
    # So S = 4.47 is safe.
    side = 4.47
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
