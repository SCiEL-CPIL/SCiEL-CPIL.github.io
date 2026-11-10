import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Set A: +/- 2 * e_i
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            points.append(v)
            
    # Set B: Vectors with exactly four non-zero entries, each +/- 1
    # Iterate over all combinations of 4 indices
    import itertools
    for combo in itertools.combinations(range(d), 4):
        # Iterate over all sign combinations for these 4 indices
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            points.append(v)
            
    # Add origin
    points.append(np.zeros(d, dtype=np.int64))
    
    arr = np.array(points, dtype=np.int64)
    return arr
