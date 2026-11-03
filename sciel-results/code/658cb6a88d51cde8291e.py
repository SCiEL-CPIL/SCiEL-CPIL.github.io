import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 3 rows of 4 hexagons in a hexagonal grid
    # Side length of inner hexagons is 1.
    # Distance between centers horizontally is sqrt(3).
    # Vertical distance between rows is 1.5.
    # Horizontal offset for odd rows is sqrt(3)/2.
    
    sqrt3 = math.sqrt(3)
    dx = sqrt3
    dy = 1.5
    dx_off = sqrt3 / 2.0
    
    # Generate positions for a 3x4 grid
    # Rows at y = -1.5, 0.0, 1.5
    # Cols at x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    # This centers the grid roughly at (0,0)
    
    xs = [-1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3, 1.5 * sqrt3]
    ys = [-1.5, 0.0, 1.5]
    
    inner = []
    for i, y in enumerate(ys):
        for x in xs:
            # All rotated 0 degrees
            inner.append([x, y, 0.0])
            
    # Outer hexagon
    # The grid spans x from -1.5*sqrt3 to 1.5*sqrt3 + 1 (apothem)
    # Max X extent approx 1.5*1.732 + 1 = 3.6
    # Max Y extent approx 1.5 + 1 = 2.5
    # For a hexagon with side S and rotation 0 (point at S,0):
    # Max X = S, Max Y = S*sqrt(3)/2
    # We need S >= 3.6 and S >= 2.5 * 2/sqrt(3) = 2.88
    # So S=4.0 is safe.
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
