import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Define the 11 centers for the hexagonal cluster
    # 1 center
    centers = [(0.0, 0.0)]
    
    # 6 in the first ring
    # Angles 0, 60, 120, 180, 240, 300
    # Distance sqrt(3)
    for i in range(6):
        angle = i * 60 * math.pi / 180.0
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        centers.append((x, y))
        
    # 4 in the second ring at specific positions to maintain symmetry and minimize bound
    # Positions at (sqrt(3), 3), (-sqrt(3), 3), (sqrt(3), -3), (-sqrt(3), -3)
    # Note: These are at distance 2*sqrt(3) from origin?
    # (sqrt(3))^2 + 3^2 = 3 + 9 = 12. sqrt(12) = 2*sqrt(3). Yes.
    centers.append((sqrt3, 3.0))
    centers.append((-sqrt3, 3.0))
    centers.append((sqrt3, -3.0))
    centers.append((-sqrt3, -3.0))
    
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon parameters
    # Side length S must contain the furthest vertex.
    # Furthest center is at distance 2*sqrt(3).
    # Vertex extends 1 unit further.
    # If outer hexagon is aligned (vertex at 60 deg), S = 2*sqrt(3) + 1.
    side = 2.0 * sqrt3 + 1.0
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
