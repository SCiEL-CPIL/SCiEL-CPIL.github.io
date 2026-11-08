import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for hexagonal packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    # We place hexagons at centers:
    # Ring 1: 6 hexagons at distance sqrt(3) from center, angles 0, 60, ...
    # Ring 2: 6 hexagons at distance 2*sqrt(3) from center, angles 0, 60, ...
    # Total 12 hexagons.
    # This configuration is symmetric and aligns with the outer hexagon.
    
    inner = []
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    for k in range(6):
        angle_rad = k * np.pi / 3.0
        x1 = d1 * np.cos(angle_rad)
        y1 = d1 * np.sin(angle_rad)
        inner.append([x1, y1, 0.0])
        
        x2 = d2 * np.cos(angle_rad)
        y2 = d2 * np.sin(angle_rad)
        inner.append([x2, y2, 0.0])
        
    # Calculate optimal side length S
    # S = 1 + max_ratio
    # For this symmetric configuration, max_ratio = 2*sqrt(3)
    S = 1.0 + 2.0 * np.sqrt(3)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
