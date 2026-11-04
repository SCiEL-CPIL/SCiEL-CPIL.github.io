"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a packing of 11 unit hexagons inside a regular hexagon of side length 3.
    The packing uses a standard hexagonal lattice arrangement.
    """
    sqrt3 = math.sqrt(3)
    
    # Centers of the 11 hexagons
    # 1 central hexagon
    centers = [(0.0, 0.0)]
    
    # 6 neighbors in the first ring (distance 1 from center)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    for i in range(6):
        angle = i * math.pi / 3.0
        x = math.cos(angle)
        y = math.sin(angle)
        centers.append((x, y))
        
    # 4 hexagons in the second ring, specifically the "radial" ones
    # aligned with the first ring axes.
    # We pick 4 of the 6 available radial positions in the second ring.
    # Positions at distance 2 from center, angles 0, 60, 120, 180.
    # Note: These are at distance 2 from origin.
    # Their vertices extend to distance 3.
    # They fit exactly in a bounding hexagon of side 3.
    for i in range(4):
        angle = i * math.pi / 3.0
        x = 2.0 * math.cos(angle)
        y = 2.0 * math.sin(angle)
        centers.append((x, y))
        
    # Construct the inner list: [x, y, angle_degrees]
    # All hexagons are aligned with the global axes (angle 0)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon parameters
    outer_center = (0.0, 0.0)
    outer_side = 3.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
