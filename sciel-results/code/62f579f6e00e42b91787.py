import numpy as np
from itertools import product, combinations

def kissing_number11() -> np.ndarray:
    d = 11
    
    # Generate candidate vectors with norm squared 4
    # Type A: one +/-2
    candidates = []
    for i in range(d):
        for s in [1, -1]:
            v = [0]*d
            v[i] = 2*s
            candidates.append(tuple(v))
            
    # Type C: four +/-1
    for support in combinations(range(d), 4):
        for signs in product([1, -1], repeat=4):
            v = [0]*d
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(tuple(v))
            
    # Greedy selection
    points = []
    
    # To ensure we get a good set, we might want to prioritize Type A?
    # They are sparse and compatible with all.
    # Let's add all Type A first.
    for v in candidates:
        if v.count(0) == d - 1: # Type A
            points.append(v)
            
    # Now add Type C
    # We can shuffle candidates to avoid bias?
    # But let's keep it deterministic first.
    
    for v in candidates:
        if v.count(0) == d - 1: # Skip Type A, already added
            continue
            
        compatible = True
        for u in points:
            # Dot product
            dot = 0
            for i in range(d):
                dot += u[i] * v[i]
            if dot > 2:
                compatible = False
                break
                
        if compatible:
            points.append(v)
            
    points_arr = np.array(points, dtype=np.int64)
    return points_arr
