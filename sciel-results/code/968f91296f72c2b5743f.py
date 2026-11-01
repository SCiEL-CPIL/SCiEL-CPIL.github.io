import itertools
import numpy as np

def construct():
    D = 11
    vectors = []
    
    # Generate all weight-4 vectors based on signatures of 3 coords
    # Signature: 3 coords, signs on them.
    # We pick 1 vector per signature.
    
    coords = range(D)
    
    for c1, c2, c3 in itertools.combinations(coords, 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[c1] = s1
                    v[c2] = s2
                    v[c3] = s3
                    
                    # Pick the 4th coordinate
                    # Must be distinct from c1, c2, c3
                    remaining = [c for c in coords if c not in (c1, c2, c3)]
                    c4 = remaining[0]
                    v[c4] = 1
                    
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
