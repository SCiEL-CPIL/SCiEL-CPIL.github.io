"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    # 3x7 grid in rectangle w=0.6, h=1.4 (perimeter = 2*(0.6+1.4) = 4)
    # r = 0.1, sum of radii = 21 * 0.1 = 2.1
    w = 0.6
    h = 1.4
    r = 0.1

    circles = []
    for i in range(3):
        for j in range(7):
            x = (i + 0.5) * w / 3.0
            y = (j + 0.5) * h / 7.0
            circles.append([x, y, r])

    return np.array(circles)
# EVOLVE-BLOCK-END
