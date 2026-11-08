"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define 11 hexagons in a compact cluster
    # 7 central hexagons (center + 6 neighbors)
    # 4 additional hexagons from the 2nd ring, chosen to minimize extent
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        # 4 additional hexes from 2nd ring (side positions, dist 3.0)
        (2, -1),
        (-2, 1),
        (1, -2),
        (-1, 2)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to cartesian
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Container parameters
    # Max extent is approx 4.0 from center (vertex of hex at dist 3.0)
    # Container side L needs to satisfy 2L >= 8.0 and sqrt(3)L >= 8.0
    # L >= 4.0 and L >= 4.62
    # Set L to 4.7 for safety margin
    side = 4.7
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
