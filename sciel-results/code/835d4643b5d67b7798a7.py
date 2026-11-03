import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Axial coordinates for 12 hexagons
    # 7 from the central cluster (side 1 hexagon of lattice)
    # 5 from the next ring (side 2 hexagon of lattice)
    # We pick 5 consecutive points from the outer ring to form a compact shape
    points_axial = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (2, 0),
        (1, 1),
        (0, 2),
        (-1, 2),
        (-2, 1)
    ]
    
    inner = []
    # Scaling factor for distance between centers is sqrt(3)
    # Conversion from axial (x, y) to Cartesian
    # x_cart = sqrt(3) * (x + y/2)
    # y_cart = 1.5 * y
    for x, y in points_axial:
        x_cart = math.sqrt(3) * (x + y / 2.0)
        y_cart = 1.5 * y
        inner.append([x_cart, y_cart, 0.0])
        
    # Outer hexagon side length
    # The centers are in a hexagon of side 2 (in lattice units).
    # Actual side of center container = 2 * sqrt(3)
    # Outer side = Center container side + 2 * inradius of inner hexagon
    # Inradius of unit hexagon = sqrt(3) / 2
    # Outer side = 2*sqrt(3) + 2*(sqrt(3)/2) = 3*sqrt(3)
    side = 3.0 * math.sqrt(3)
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
