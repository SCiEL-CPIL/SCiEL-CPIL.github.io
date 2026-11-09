# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49, p = 7.
    
    F_49 = F_7[alpha], alpha^2 = 3. Point (x,y) written as [[xa,xb],[ya,yb]].
    
    Construction: C = x-axis ∪ y-axis (the "cross").
    C = {(a, 0) : a in F_49} ∪ {(0, b) : b in F_49}
    |C| = 49 + 49 - 1 = 97 (intersection at origin counted once).
    N = F_49^2 \\ C, |N| = 2401 - 97 = 2304.
    
    Nikodym validity:
    - For x = (a, 0) on x-axis, a != 0: vertical line x = a has C ∩ L = {(a,0)} = {x}. ✓
    - For x = (0, b) on y-axis, b != 0: horizontal line y = b has C ∩ L = {(0,b)} = {x}. ✓
    - For x = (0, 0) origin: any line with slope m != 0 has C ∩ L = {(0,0)} = {x}. ✓
    - For n = (a, b) in N (a != 0 and b != 0): 
      * vertical line x = a: C ∩ L = {(a, 0)}. L \\ {n} ⊆ N since the only C point is (a,0) != n. ✓
      * (also horizontal line y = b: C ∩ L = {(0, b)}, works similarly)
    
    Score = |C| / 147 = 97/147 ≈ 0.660.
    """
    p = 7
    q = p * p  # 49
    
    # F_49 elements are pairs (a, b) representing a + b*alpha, a,b in 0..6
    # x-axis: y = (0, 0), x = (xa, xb) for all xa, xb in 0..6
    # y-axis: x = (0, 0), y = (ya, yb) for all ya, yb in 0..6
    
    complement = []  # C = x-axis ∪ y-axis
    
    for xa in range(p):
        for xb in range(p):
            # x-axis point: (x, (0,0))
            complement.append([[xa, xb], [0, 0]])
    
    for ya in range(p):
        for yb in range(p):
            # y-axis point: ((0,0), y)
            # Skip origin to avoid duplicate
            if ya == 0 and yb == 0:
                continue
            complement.append([[0, 0], [ya, yb]])
    
    # C has 49 + 48 = 97 points
    # N = F_49^2 \\ C
    C_set = set()
    for pt in complement:
        C_set.add((pt[0][0], pt[0][1], pt[1][0], pt[1][1]))
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in C_set:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
