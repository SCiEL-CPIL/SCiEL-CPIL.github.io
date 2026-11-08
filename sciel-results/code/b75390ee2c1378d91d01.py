import math

def construct():
    # N = 11
    # Centers of the 7-hexagon core
    sqrt3 = math.sqrt(3)
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5),
        (-sqrt3/2, -1.5)
    ]
    
    # 4 additional hexagons at the "corners" of the next shell
    # Axial (1,1) -> Cartesian (1.5*sqrt3, 1.5)
    # Axial (1,-1) -> Cartesian (1.5*sqrt3, -1.5)
    # Axial (-1,1) -> Cartesian (-1.5*sqrt3, 1.5)
    # Axial (-1,-1) -> Cartesian (-1.5*sqrt3, -1.5)
    extra_centers = [
        (1.5*sqrt3, 1.5),
        (1.5*sqrt3, -1.5),
        (-1.5*sqrt3, 1.5),
        (-1.5*sqrt3, -1.5)
    ]
    
    all_centers = centers + extra_centers
    
    inner = []
    for x, y in all_centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side length 4.0
    # Angle 30 degrees (so vertices are at 30, 90, 150, 210, 270, 330)
    # This aligns the vertices with the furthest points of the inner hexagons
    outer_center = (0.0, 0.0)
    outer_side = 4.0
    outer_angle = 30.0
    
    return inner, outer_center, outer_side, outer_angle
