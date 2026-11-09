# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a minimal Nikodym set in F_49^2.
    
    Construction: Let C = {(t, 0) : t in F_49} be the "x-axis" (y=0).
    Let N = F_49^2 \\ C, i.e., all points with y != (0,0).
    |N| = 49 * 48 = 2352, so complement size = 49.
    
    Verification:
    - For x in C (y=0): the vertical line through x (direction (0,1)) has all
      other points with y != 0, hence in N.
    - For x in N (y != 0): the horizontal line through x (direction (1,0)) has
      all points with the same y != 0, hence all in N.
    """
    p = 7
    # F_49 elements: (a, b) representing a + b*alpha, a,b in 0..6
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Skip points on C: y = (0, 0)
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
# EVOLVE-BLOCK-END
