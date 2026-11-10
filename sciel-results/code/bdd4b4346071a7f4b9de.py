import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # S is the line y = 0 in F_49^2
    # A point (x, y) is in S if y == 0 (i.e., y_a == 0 and y_b == 0)
    # N is the complement of S
    
    points = []
    for x_a in range(p):
        for x_b in range(p):
            for y_a in range(p):
                for y_b in range(p):
                    # Check if (x, y) is in S (y = 0)
                    if y_a == 0 and y_b == 0:
                        continue
                    points.append([[x_a, x_b], [y_a, y_b]])
    
    return np.array(points, dtype=np.int64)
