import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 4 columns, 3 rows of Pointy-Top hexagons
    # Pointy-Top: Vertex at top (angle 90)
    # Grid spacing:
    # x-step: sqrt(3)
    # y-step: 1.5
    # Offset for odd rows: sqrt(3)/2
    
    sqrt3 = math.sqrt(3)
    inner = []
    
    # Center the grid
    # i in 0..3, j in 0..2
    for j in range(3):
        y = (j - 1) * 1.5
        for i in range(4):
            # Base x
            x_base = (i - 1.5) * sqrt3
            # Offset for odd j (1)
            if j % 2 == 1:
                x = x_base + sqrt3 / 2
            else:
                x = x_base
            inner.append([x, y, 90.0])
            
    # Outer hexagon
    # Pointy-Top, Side 4.0
    center = (0.0, 0.0)
    side = 4.0
    angle = 90.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
