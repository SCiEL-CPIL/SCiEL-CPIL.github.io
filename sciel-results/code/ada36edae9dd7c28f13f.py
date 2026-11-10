# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    
    p = 7
    q = 49  # |F_49|
    
    # S = F_49 x C where C = F_49 \ {c0} for some fixed c0.
    # |S| = 49 * 48 = 2352, |N| = 49.
    # 
    # Validity:
    # - x in S (y in C): 
    #   * dir (0,1): vertical line, hits all y, so hits 48 other S-points. OK.
    #   * dir (1,m), m!=0: line (x+t, y+tm). Hits y=c' in C when t=(c'-y)/m.
    #     For each c' in C, c'!=y: one t value. 47 other S-points. OK.
    #   * dir (1,m), m=0: line (x+t, y). All y, so all 49 points in S. OK.
    # - x not in S (y=c0):
    #   * dir (1,0): line (x+t, c0). y=c0 not in C, so no S-points. Valid witness!
    #
    # So N = {x : y = c0} is a valid Nikodym set with |N| = 49.
    
    # Choose c0 = 0 (the zero element, index 0 in F_49)
    c0 = 0
    
    # N = all points with y = c0
    N_points = []
    for x in range(q):
        xa, xb = x % 7, x // 7
        ya, yb = c0 % 7, c0 // 7
        N_points.append([[xa, xb], [ya, yb]])
    
    result = np.array(N_points, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
