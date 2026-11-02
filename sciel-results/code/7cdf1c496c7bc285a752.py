import numpy as np

def construct():
    """
    Places 11 disjoint regular hexagons of side length 1 inside a regular hexagon,
    minimizing the outer side length.
    """
    # Inner hexagons: list of (x, y, angle_degrees)
    # We use a packing with 1 central, 6 surrounding, and 4 more in the gaps.
    # To fit them in a smaller outer hexagon, we rotate the inner hexagons by 30 degrees.
    # This reduces their radial extent to the apothem (sqrt(3)/2) instead of the circumradius (1).
    
    inner = []
    
    # 1. Center hexagon
    inner.append((0.0, 0.0, 30.0))
    
    # 2-7. 6 neighbors at distance sqrt(3) from center
    # Angles: 0, 60, 120, 180, 240, 300
    for i in range(6):
        angle_deg = i * 60.0
        angle_rad = np.deg2rad(angle_deg)
        x = np.sqrt(3) * np.cos(angle_rad)
        y = np.sqrt(3) * np.sin(angle_rad)
        inner.append((x, y, 30.0))
        
    # 8-11. 4 more hexagons in the gaps (angles 30, 90, 150, 210) at distance 3
    # These fit because with 30 deg rotation, radial extent is sqrt(3)/2 ~ 0.866
    # Total radial reach = 3 + 0.866 = 3.866
    # Outer hexagon side 4.5 has apothem 4.5 * sqrt(3)/2 ~ 3.897
    # 3.866 < 3.897, so they fit.
    gap_angles = [30.0, 90.0, 150.0, 210.0]
    for angle_deg in gap_angles:
        angle_rad = np.deg2rad(angle_deg)
        x = 3.0 * np.cos(angle_rad)
        y = 3.0 * np.sin(angle_rad)
        inner.append((x, y, 30.0))
        
    # Outer hexagon parameters
    center = (0.0, 0.0)
    side = 4.5
    angle = 0.0
    
    return inner, center, side, angle
