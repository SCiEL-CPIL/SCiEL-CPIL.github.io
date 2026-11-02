import numpy as np

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Uses a hexagonal packing pattern with 4 rows (6, 5, 5, 5 circles).
    """
    n = 21
    # Hexagonal packing with 4 rows: 6, 5, 5, 5
    # Width w = 12r
    # Height h = (2 + 3*sqrt(3))*r
    # Perimeter: 2*(w+h) = 4 => w+h = 2
    # (12 + 2 + 3*sqrt(3))*r = 2
    # r = 2 / (14 + 3*sqrt(3))
    
    sqrt3 = np.sqrt(3.0)
    denom = 14.0 + 3.0 * sqrt3
    r = 2.0 / denom
    
    w = 12.0 * r
    h = (2.0 + 3.0 * sqrt3) * r
    
    # Verify perimeter
    # print(f"r={r}, w={w}, h={h}, perimeter={2*(w+h)}")
    
    circles = []
    
    # Row 0: 6 circles, y = r, x = r, 3r, 5r, 7r, 9r, 11r
    for i in range(6):
        x = r + 2.0 * i * r
        y = r
        circles.append((x, y, r))
        
    # Row 1: 5 circles, y = r + sqrt(3)*r, x = 2r, 4r, 6r, 8r, 10r
    for i in range(5):
        x = 2.0 * r + 2.0 * i * r
        y = r + sqrt3 * r
        circles.append((x, y, r))
        
    # Row 2: 5 circles, y = r + 2*sqrt(3)*r, x = r, 3r, 5r, 7r, 9r
    for i in range(5):
        x = r + 2.0 * i * r
        y = r + 2.0 * sqrt3 * r
        circles.append((x, y, r))
        
    # Row 3: 5 circles, y = r + 3*sqrt(3)*r, x = 2r, 4r, 6r, 8r, 10r
    for i in range(5):
        x = 2.0 * r + 2.0 * i * r
        y = r + 3.0 * sqrt3 * r
        circles.append((x, y, r))
        
    return np.array(circles[:n])

# EVOLVE-BLOCK-END
