import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate all vectors with 4 non-zero coordinates (+/- 1)
    # There are C(11, 4) * 2^4 = 330 * 16 = 5280 such vectors.
    vectors = []
    for idx in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(idx, signs):
                v[i] = s
            vectors.append(v)
    
    vectors = np.array(vectors)
    
    # We want a subset where min pairwise distance squared >= 4.
    # Equivalent to max pairwise dot product <= 2 (since norm sq = 4).
    
    # Greedy selection
    selected = []
    # Shuffle to avoid bias
    np.random.seed(42)
    indices = np.random.permutation(len(vectors))
    
    for i in indices:
        if time.time() > deadline:
            break
        v = vectors[i]
        # Check against all selected
        if not selected:
            selected.append(v)
            continue
            
        # Compute dot products with all selected
        # To speed up, we can break early if dot > 2
        compatible = True
        for u in selected:
            if np.dot(v, u) > 2:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
            
    if not selected:
        # Fallback to seed if something went wrong
        return np.array([[0]*D], dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
