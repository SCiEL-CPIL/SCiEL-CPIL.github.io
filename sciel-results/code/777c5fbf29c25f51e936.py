import numpy as np

def construct():
    # Use a 3x7 grid in a rectangle with perimeter 4
    # This allows a radius of 0.1, sum = 2.1
    # Seed sum was ~2.079
    
    m = 3
    k = 7
    # Diameter D
    # W = m*D, H = k*D
    # 2(W+H) = 4 => W+H=2
    # D(m+k) = 2 => D = 2/(m+k)
    D = 2.0 / (m + k)
    r = D / 2.0
    
    W = m * D
    H = k * D
    
    circles = []
    for i in range(m):
        for j in range(k):
            x = (i + 0.5) * D
            y = (j + 0.5) * D
            circles.append([x, y, r])
            
    return np.array(circles[:21])
