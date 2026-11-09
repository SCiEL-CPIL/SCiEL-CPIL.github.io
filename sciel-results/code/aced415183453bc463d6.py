import time
import random
from itertools import combinations

def _independence_number_fastest(triples):
    if not triples:
        return 16
        
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
        
    max_alpha = 0
    # Iterate all 2^16 masks
    # We can prune: if popcount(mask) <= max_alpha, skip?
    # No, we need to find the max.
    # But we can check larger masks first?
    # Iterating 0 to 2^16-1 doesn't order by popcount.
    # We can just compute max.
    
    for mask in range(1 << 16):
        # Quick check: if popcount <= max_alpha, we can't improve, but we still need to check if it's independent?
        # No, if it's not independent, it doesn't matter.
        # If it IS independent, and popcount > max_alpha, we update.
        # So we can skip the independence check if popcount <= max_alpha?
        # Yes! If popcount <= max_alpha, even if it is independent, it won't improve the result.
        if bin(mask).count('1') <= max_alpha:
            continue
            
        is_ind = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_ind = False
                break
        if is_ind:
            max_alpha = bin(mask).count('1')
            
    return max_alpha

def _generate_random_packing():
    all_triples = list(combinations(range(16), 3))
    random.shuffle(all_triples)
    
    packing = []
    for t in all_triples:
        t_set = set(t)
        valid = True
        for existing in packing:
            if len(t_set.intersection(existing)) > 1:
                valid = False
                break
        if valid:
            packing.append(list(t))
            
    return packing

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_packing = []
    best_alpha = 17
    
    while time.time() < deadline:
        packing = _generate_random_packing()
        alpha = _independence_number_fastest(packing)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_packing = [list(t) for t in packing]
            
    return best_packing
