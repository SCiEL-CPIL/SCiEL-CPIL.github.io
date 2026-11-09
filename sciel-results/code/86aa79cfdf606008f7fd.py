import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    sqrt3 = math.sqrt(3.0)
    
    # Define the 12 inner hexagons
    # We use a subset of the hexagonal grid.
    # 7 hexes from the k=2 cluster (center + 6 neighbors)
    # 5 hexes from the k=3 cluster corners that fit in L=4
    
    inner = []
    
    # Helper to convert axial (q, r) to cartesian (x, y) for flat-top hexagons
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    # 1. Central hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. The 6 neighbors (k=2 cluster)
    neighbors = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1)
    ]
    for q, r in neighbors:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    # 3. Add 5 hexes from the next ring (k=3) that fit in L=4
    # The k=3 ring has 12 hexes. 
    # The ones that fit in L=4 are those whose centers are within the shrunk hex (side 3).
    # These correspond to the 6 "corner" positions of the k=3 hexagon shape, 
    # specifically the ones aligned with the vertices of the outer hex? 
    # Actually, the 6 hexes at (2,0), (2,-1), (1,1), (-1,2), (-2,1), (-2,0) fit.
    # Let's verify (1,1): x=1.5, y=1.5*sqrt3. Dist = 3.
    # The shrunk hex has radius 3. So these 6 fit.
    # We need 5 more. We pick any 5 of these 6.
    
    extra_hexes = [
        (2, 0),
        (2, -1),
        (1, 1),
        (-1, 2),
        (-2, 1)
    ]
    
    for q, r in extra_hexes:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    # Total 1 + 6 + 5 = 12 hexagons.
    
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
