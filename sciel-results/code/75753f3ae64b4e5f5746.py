import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all vectors with exactly two non-zero entries, which are +/- 1
    # Norm squared is 2.
    # Min squared distance between any two such distinct vectors is >= 2.
    # Check:
    # If they share 0 indices: dist^2 = 2+2 = 4.
    # If they share 1 index:
    #   Same sign: e.g. (1,1,0) and (1,0,1). diff=(0,1,-1). dist^2=2.
    #   Diff sign: e.g. (1,1,0) and (-1,0,1). diff=(2,1,-1). dist^2=6.
    # If they share 2 indices:
    #   They are either identical or differ in sign of one/both.
    #   (1,1,0) and (1,-1,0): diff=(0,2,0). dist^2=4.
    #   (1,1,0) and (-1,-1,0): diff=(2,2,0). dist^2=8.
    #   (1,1,0) and (-1,1,0): diff=(2,0,0). dist^2=4.
    # All >= 2.
    # Max norm squared is 2.
    # Condition: min_dist^2 >= max_norm^2 => 2 >= 2. Holds.
    
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    points.append(v)
    
    result = np.array(points)
    return result
