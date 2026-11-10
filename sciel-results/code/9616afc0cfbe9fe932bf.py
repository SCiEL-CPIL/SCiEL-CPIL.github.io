import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    vectors = []
    
    # Generate all vectors with exactly two non-zero entries, which are +/- 1
    # There are C(11, 2) pairs of indices, and 4 sign combinations for each pair
    for i in range(d):
        for j in range(i + 1, d):
            # Signs: (1, 1), (1, -1), (-1, 1), (-1, -1)
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    vec = np.zeros(d, dtype=np.int64)
                    vec[i] = s1
                    vec[j] = s2
                    vectors.append(vec)
                    
    return np.array(vectors)
