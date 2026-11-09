import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Time guard
    start_time = time.time()
    deadline = 600 - 10  # 590s
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Elements are (a, b) representing a + b*alpha
    # We need to map (a, b) to a unique integer or just use the tuple.
    # The output format is [[x_a, x_b], [y_a, y_b]]
    
    # S is the set of points (x, 0) where x in F_49.
    # (x, 0) means y_a = 0, y_b = 0.
    # x can be any (x_a, x_b) in F_7 x F_7.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if (x, y) is in S
                    # S = { (x, 0) } => y_a == 0 and y_b == 0
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return np.array(points, dtype=np.int64)
