import itertools
import numpy as np

def construct():
    D = 11
    # Generate all integer vectors with exactly three non-zero entries of +/- 1
    # These have norm sqrt(3)
    vectors = []
    for idxs in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[idxs[0]] = s1
                    v[idxs[1]] = s2
                    v[idxs[2]] = s3
                    vectors.append(np.array(v))
    
    # Greedy selection
    selected = []
    for v in vectors:
        valid = True
        for u in selected:
            dist_sq = np.sum((v - u) ** 2)
            if dist_sq < 3:
                valid = False
                break
        if valid:
            selected.append(v)
            
    return np.array(selected)
