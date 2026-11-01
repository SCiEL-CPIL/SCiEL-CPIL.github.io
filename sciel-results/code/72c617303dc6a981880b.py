import math

def construct():
    # 11 unit hexagons
    # Centers in Cartesian coordinates
    # Inner 7 (Radius 1 cluster)
    r1 = math.sqrt(3)
    centers = [
        (0.0, 0.0),
        (0.0, r1),
        (r1/2, r1/2),
        (r1/2, -r1/2),
        (0.0, -r1),
        (-r1/2, -r1/2),
        (-r1/2, r1/2)
    ]
    
    # 4 Outer (Radius 2, Side positions only)
    # Side 1 (30 deg): (1.5*sqrt(3), 1.5)
    # Side 3 (150 deg): (-1.5*sqrt(3), 1.5)
    # Side 4 (210 deg): (-1.5*sqrt(3), -1.5)
    # Side 6 (330 deg): (1.5*sqrt(3), -1.5)
    
    s = 1.5 * math.sqrt(3)
    outer_4 = [
        (s, 1.5),
        (-s, 1.5),
        (-s, -1.5),
        (s, -1.5)
    ]
    
    centers.extend(outer_4)
    
    # Format inner: [[x, y, angle_degrees], ...]
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    # Center (0,0)
    # Side length 4.0
    # Angle 30.0 degrees (Pointy top)
    # This aligns the outer vertices with the radial directions of the 4 outer inner hexagons.
    
    return inner, (0.0, 0.0), 4.0, 30.0
