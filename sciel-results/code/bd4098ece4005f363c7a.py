import numpy as np

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Perturb angles to break symmetry at close range, preserve at far range
    delta = 0.05
    eps = np.zeros(n)
    # Set eps for i=1..8, 10, 12..19 to delta
    # Keep eps for i=9, 11 at 0
    for i in range(n):
        if i == 9 or i == 11:
            eps[i] = 0.0
        else:
            eps[i] = delta
            
    angles = angles + eps
    
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Ensure strictly convex (should be with small delta)
    return polygon
