import numpy as np

def construct():
    N = 21
    # 7 rows of 3 circles
    # Hexagonal packing
    # Width W = 7r
    # Height H = (2 + 6*sqrt(3))r
    # Perimeter 2(W+H) <= 4 => W+H = 2
    # r * (9 + 6*sqrt(3)) = 2
    
    sqrt3 = np.sqrt(3)
    coeff = 9 + 6 * sqrt3
    r = 2.0 / coeff
    
    centers = []
    # 7 rows
    for i in range(7):
        y = r + i * sqrt3 * r
        # Alternate shift to minimize width
        # Shift 0: x = r, 3r, 5r
        # Shift 1: x = 2r, 4r, 6r
        if i % 2 == 0:
            xs = [r, 3*r, 5*r]
        else:
            xs = [2*r, 4*r, 6*r]
        for x in xs:
            centers.append([x, y, r])
            
    return np.array(centers)
