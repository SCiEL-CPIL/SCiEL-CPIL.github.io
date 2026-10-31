import numpy as np

def construct():
    N = 21
    m = 7
    k = 3
    
    # Calculate max radius
    # w = 2 * k * r
    # h = (2 + (m-1)*sqrt(3)) * r
    # 2*w + 2*h <= 4  => w + h <= 2
    # r * (2*k + 2 + (m-1)*sqrt(3)) <= 2
    sqrt3 = np.sqrt(3)
    denom = 2 * k + 2 + (m - 1) * sqrt3
    r_max = 2.0 / denom
    
    # Add small epsilon to ensure disjoint interiors and perimeter constraint
    r = r_max * (1 - 1e-6)
    
    circles = []
    for i in range(m):
        for j in range(k):
            # x coordinate
            # Row i is offset by i*r if i is odd?
            # Standard hex packing:
            # Even rows: x = r + j * 2r
            # Odd rows: x = 2r + j * 2r
            # y coordinate: y = r + i * sqrt(3) * r
            
            if i % 2 == 0:
                x = r + j * 2 * r
            else:
                x = 2 * r + j * 2 * r
            
            y = r + i * sqrt3 * r
            
            circles.append([x, y, r])
            
    return np.array(circles)
