import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Part 1: 2 * E8 vectors (first 8 dimensions)
    # Type A: Two +/-2, rest 0
    for i, j in itertools.combinations(range(8), 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    # Type B: Eight +/-1, rest 0, sum of coords divisible by 4
    # Signs are in {1, -1}
    # We iterate over all 2^8 sign combinations
    # and check if sum is divisible by 4
    for signs in itertools.product([1, -1], repeat=8):
        s_sum = sum(signs)
        if s_sum % 4 == 0:
            v = np.zeros(D, dtype=np.int64)
            for k, s in enumerate(signs):
                v[k] = s
            vectors.append(v)

    # Part 2: 2 * D3 vectors (last 3 dimensions, indices 8, 9, 10)
    # Two +/-2, rest 0 in the last 3 positions
    for i, j in itertools.combinations(range(8, 11), 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)

    return np.array(vectors, dtype=np.int64)
