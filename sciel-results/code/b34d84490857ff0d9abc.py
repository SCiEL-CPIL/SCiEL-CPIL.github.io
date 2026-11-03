import numpy as np
from itertools import combinations

def construct():
    D = 11
    vectors = []
    
    # Generate all 4-subsets of {0, ..., 10}
    all_subsets = list(combinations(range(D), 4))
    
    # Greedy selection of subsets such that no 3 elements are shared
    # This ensures intersection size <= 2
    used_3_subsets = set()
    selected_subsets = []
    
    # Shuffle to avoid bias? Or just iterate.
    # A specific order might yield better packing?
    # Let's try random shuffling a few times and keep the best?
    # Given 120s limit, we can try a few times.
    
    best_count = 0
    best_selected = []
    
    import random
    
    for _ in range(10):
        random.shuffle(all_subsets)
        used = set()
        selected = []
        
        for subset in all_subsets:
            # Get all 3-subsets of this 4-subset
            subs = list(combinations(subset, 3))
            
            # Check if any is already used
            if any(s in used for s in subs):
                continue
            
            # Add to selected
            selected.append(subset)
            # Mark 3-subsets as used
            for s in subs:
                used.add(s)
                
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
            
    # Generate vectors from best_selected
    for subset in best_selected:
        i, j, k, l = subset
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        vectors.append(v)
                        
    return np.array(vectors, dtype=np.int64)
