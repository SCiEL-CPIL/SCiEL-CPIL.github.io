import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all k-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        
    # We need to map mask back to sorted list of elements for the output
    # But we can just return the masks converted to lists at the end.
    # Or keep the original tuples.
    # Let's keep tuples for easy conversion.
    set_tuples = list(combinations(range(1, n+1), k))
    # Map tuple to mask
    tuple_to_mask = {}
    for t in set_tuples:
        mask = 0
        for x in t:
            mask |= (1 << (x-1))
        tuple_to_mask[t] = mask
        
    # We will work with indices into set_tuples for speed?
    # Or just work with masks and convert at the end.
    # Converting mask to tuple:
    def mask_to_tuple(mask):
        res = []
        for i in range(n):
            if mask & (1 << i):
                res.append(i+1)
        return tuple(res)
        
    best_family = []
    best_size = 0
    
    # Run multiple greedy restarts
    # Estimate time per restart: ~1-2s
    # We can run ~20-30 restarts.
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(restart)
        shuffled_indices = list(range(len(set_tuples)))
        random.shuffle(shuffled_indices)
        
        family_masks = []
        # Pre-compute intersections? No, just check on fly.
        
        for idx in shuffled_indices:
            s_mask = tuple_to_mask[set_tuples[idx]]
            
            # Check if s_mask can be added
            valid = True
            # Check against all pairs in family_masks
            # Optimization: if family is large, this is slow.
            # But 250^2 / 2 = 31250 checks.
            # 1820 candidates * 31250 = 57M checks.
            # In Python, 57M simple ops might take 5-10s.
            # We might only get 5-10 restarts.
            
            for i in range(len(family_masks)):
                a = family_masks[i]
                ia = a & s_mask
                for j in range(i+1, len(family_masks)):
                    b = family_masks[j]
                    ib = b & s_mask
                    if ia == ib:
                        if (a & b) == ia:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                family_masks.append(s_mask)
                
        if len(family_masks) > best_size:
            best_size = len(family_masks)
            best_family = family_masks[:]
            
    # Convert best_family masks to sorted lists
    result = []
    for mask in best_family:
        result.append(list(mask_to_tuple(mask)))
        
    return result
