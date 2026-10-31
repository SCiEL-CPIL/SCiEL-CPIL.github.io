import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_vectors = []
    best_score = 0
    
    # We try different weights w
    # w=1 is trivial, w=2 is seed (220)
    # We focus on w >= 3
    
    for w in range(3, D + 1):
        if time.time() > deadline:
            break
            
        # Max allowed intersection size
        max_k = w // 2
        
        # Generate all supports
        supports = list(itertools.combinations(range(D), w))
        
        # Greedy selection of supports
        selected_supports = []
        for s in supports:
            # Check against all previously selected
            valid = True
            for s_sel in selected_supports:
                # Intersection size
                inter_size = len(set(s) & set(s_sel))
                if inter_size > max_k:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
                
        if not selected_supports:
            continue
            
        # Generate vectors for each selected support
        # We use even parity signs to ensure min Hamming distance 1 (which is min dist 2 in vector space? No, HD 1 -> dist sqrt(4)=2)
        # Wait, for w=5, we need HD >= 2.
        # Even parity set has min HD 2?
        # In Q_n, the bipartition by parity:
        # Distance between two even parity vectors is even.
        # Min non-zero even distance is 2.
        # So even parity set has min HD 2.
        # This satisfies HD >= 2 for w=5.
        # For w=4, we need HD >= 1. Even parity has min HD 2, which is >= 1. So it works.
        # For w=3, we need HD >= 1. Even parity has min HD 2. Works.
        
        vectors = []
        for s in selected_supports:
            # Generate all sign combinations
            # We only take those with even number of -1s (or +1s? Parity is relative)
            # Let's just take all 2^w and filter for even parity of -1s.
            # Or simpler: iterate all 2^w, count -1s, if even, keep.
            for signs in itertools.product([1, -1], repeat=w):
                if signs.count(-1) % 2 == 0:
                    v = [0] * D
                    for idx, val in zip(s, signs):
                        v[idx] = val
                    vectors.append(v)
                    
        score = len(vectors)
        if score > best_score:
            best_score = score
            best_vectors = vectors
            
    if not best_vectors:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = vectors

    return np.array(best_vectors, dtype=np.int64)
