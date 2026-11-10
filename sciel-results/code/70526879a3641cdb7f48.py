import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all weight-4 vectors
    vectors = []
    for indices in itertools.combinations(range(d), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
    
    # We need to select a subset such that no two vectors share a signed triple with the same signs.
    # A signed triple is (i, j, k, s_i, s_j, s_k) with i<j<k.
    
    # Precompute the triples for each vector
    vector_triples = []
    for v in vectors:
        triples = []
        # Find non-zero indices
        non_zero = np.nonzero(v)[0]
        # There should be exactly 4
        for comb in itertools.combinations(non_zero, 3):
            # Sort indices to make key unique
            i, j, k = sorted(comb)
            si = v[i]
            sj = v[j]
            sk = v[k]
            triples.append((i, j, k, si, sj, sk))
        vector_triples.append(triples)
        
    # Greedy packing
    used_triples = set()
    selected_vectors = []
    
    # Shuffle to get a random valid packing
    indices = list(range(len(vectors)))
    np.random.shuffle(indices)
    
    for idx in indices:
        triples = vector_triples[idx]
        # Check if any triple is already used
        if all(t not in used_triples for t in triples):
            selected_vectors.append(vectors[idx])
            for t in triples:
                used_triples.add(t)
                
    if not selected_vectors:
        # Fallback to weight-2 if packing fails (unlikely)
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
        return np.array(points)

    return np.array(selected_vectors)
