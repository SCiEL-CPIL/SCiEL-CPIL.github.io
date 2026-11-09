import math

def construct():
    # Lattice vectors for flat-top hexagons of side 1
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    sqrt3 = math.sqrt(3)
    
    def hex_coord(i, j):
        x = sqrt3 * (i + j / 2.0)
        y = 1.5 * j
        return x, y

    # Define the 12 hexagons
    # 1 central
    coords = [(0, 0)]
    
    # 6 neighbors (ring 1)
    neighbors = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1)
    ]
    for i, j in neighbors:
        coords.append(hex_coord(i, j))
        
    # 5 from ring 2 (choosing 5 of the 6 available at distance 3)
    # The 6 available are at (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)
    # We exclude (-1, 2) which is at angle 90 deg (top)
    ring2_subset = [
        (1, 1),   # 30 deg
        (-1, -1), # 210 deg
        (2, -1),  # 330 deg (-30)
        (-2, 1),  # 150 deg
        (1, -2)   # 270 deg (-90)
    ]
    for i, j in ring2_subset:
        coords.append(hex_coord(i, j))

    inner = []
    for x, y in coords:
        inner.append([x, y, 0.0])

    # Outer hexagon
    # Center (0,0)
    # Side length 4.0
    # Angle 30.0 degrees to align vertices with the 5 outer hexagons' furthest vertices
    center = (0.0, 0.0)
    side = 4.0
    angle = 30.0

    return inner, center, side, angle
