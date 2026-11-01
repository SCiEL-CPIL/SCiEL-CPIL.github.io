import math

def construct():
    # Define the 13 hexagons in a hexagonal lattice
    # Axial coordinates (q, r)
    # Center (0,0)
    # Neighbors (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Next ring (distance 3): (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0)
    
    # Convert axial to pixel (x, y) for unit hexagon
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    hexes = []
    # Add center
    hexes.append((0.0, 0.0, 0.0))
    
    # Add 6 neighbors
    neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for q, r in neighbors:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        hexes.append((x, y, 0.0))
        
    # Add 6 from next ring
    next_ring = [(2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0)]
    for q, r in next_ring:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        hexes.append((x, y, 0.0))
        
    # We have 13 hexes. We need 11.
    # Take the first 11.
    inner = hexes[:11]
    
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.5
    angle = 0.0
    
    return inner, center, side, angle
