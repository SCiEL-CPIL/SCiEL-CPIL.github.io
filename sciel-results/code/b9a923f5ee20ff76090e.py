import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s budget
    
    all_sets = list(combinations(range(1, 17), 4))
    n_sets = len(all_sets)
    
    # Precompute intersection sizes between all pairs
    # Use bitmasks for speed
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    masks = [to_mask(s) for s in all_sets]
    
    # Precompute pairwise intersection sizes
    # 1820 x 1820 is about 3.3M entries, manageable
    # But let's be smart: only compute what we need
    
    def inter_size(i, j):
        return bin(masks[i] & masks[j]).count('1')
    
    # For a set i, given current chosen set C, check if adding i creates a sunflower
    # with any pair in C. A sunflower with i and two sets a, b in C means
    # inter(i,a) == inter(i,b) == inter(a,b)
    
    # Greedy with randomization and multiple passes
    best_family = []
    
    def is_valid_add(candidate_idx, chosen_indices):
        """Check if adding candidate to chosen doesn't create a sunflower."""
        ci = candidate_idx
        # For each pair in chosen, check if (ci, a, b) forms a sunflower
        for a in chosen_indices:
            ia = inter_size(ci, a)
            for b in chosen_indices:
                if b <= a:
                    continue
                ib = inter_size(ci, b)
                if ia != ib:
                    continue
                ab = inter_size(a, b)
                if ia == ab:
                    return False
        return True
    
    # Strategy: multiple randomized greedy passes
    # Order by random permutation, greedily add
    best_size = 0
    best_result = []
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random permutation
        order = list(range(n_sets))
        random.shuffle(order)
        
        chosen = []
        chosen_set = set()
        
        for idx in order:
            if time.time() > deadline:
                break
            if is_valid_add(idx, chosen):
                chosen.append(idx)
                chosen_set.add(idx)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_result = [all_sets[i] for i in chosen]
    
    # Also try a deterministic greedy with different orderings
    if time.time() < deadline:
        # Try ordering by bitmask value
        order = sorted(range(n_sets), key=lambda i: masks[i])
        chosen = []
        for idx in order:
            if time.time() > deadline:
                break
            if is_valid_add(idx, chosen):
                chosen.append(idx)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_result = [all_sets[i] for i in chosen]
    
    if time.time() < deadline:
        # Try reverse order
        order = sorted(range(n_sets), key=lambda i: masks[i], reverse=True)
        chosen = []
        for idx in order:
            if time.time() > deadline:
                break
            if is_valid_add(idx, chosen):
                chosen.append(idx)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_result = [all_sets[i] for i in chosen]
    
    if not best_result:
        # Fallback: just return empty or minimal
        return []
    
    return [sorted(s) for s in best_result]
