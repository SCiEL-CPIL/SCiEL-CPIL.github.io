import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    centers = []
    
    # Center hexagon
    centers.append((0.0, 0.0))
    
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * math.pi / 3.0
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        centers.append((x, y))
        
    # Second ring (4 hexagons, extending the first 4 arms)
    for i in range(4):
        angle = i * math.pi / 3.0
        x = 2.0 * sqrt3 * math.cos(angle)
        y = 2.0 * sqrt3 * math.sin(angle)
        centers.append((x, y))
        
    # Format inner hexagons: [x, y, angle_degrees]
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    # The max distance from center to a tip is 2*sqrt(3) + 1
    outer_center = (0.0, 0.0)
    outer_side = 2.0 * sqrt3 + 1.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
