import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 3 rows of 4 hexagons
    # Pointy-top orientation (angle 0)
    # Width of hex = sqrt(3), Height = 2
    # Horizontal spacing = sqrt(3)
    # Vertical spacing = 1.5
    # Staggering = sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    inner = []
    
    # Row 0: y = -1.5
    # 4 hexagons, staggered left
    x_offsets = [-2*sqrt3, -sqrt3, 0, sqrt3]
    for x in x_offsets:
        inner.append([x, -1.5, 0.0])
        
    # Row 1: y = 0.0
    # 4 hexagons, centered
    x_offsets = [-1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3]
    for x in x_offsets:
        inner.append([x, 0.0, 0.0])
        
    # Row 2: y = 1.5
    # 4 hexagons, staggered right
    x_offsets = [-sqrt3, 0, sqrt3, 2*sqrt3]
    for x in x_offsets:
        inner.append([x, 1.5, 0.0])
        
    # Container
    # Max x extent: 2.5*sqrt3 ~ 4.33 at y=1.5
    # Constraint for pointy-top hexagon side S:
    # x <= sqrt(3) * (S - y) for y > 0
    # 4.33 <= sqrt(3) * (S - 1.5)
    # 2.5 <= S - 1.5
    # S >= 4.0
    
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
