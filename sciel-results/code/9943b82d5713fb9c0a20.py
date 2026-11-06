import random
import time
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n = len(all_sets)  # 1820
    
    best_size = 0
    best_family = []
    
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for overhead
    
    # Helper to check if c can be added to chosen
    # chosen is a list of masks
    def can_add(c, chosen):
        buckets = {}
        for a in chosen:
            inter = a & c
            if inter not in buckets:
                buckets[inter] = []
            buckets[inter].append(a)
        
        for y, group in buckets.items():
            if len(group) < 2:
                continue
            g_len = len(group)
            # Check pairs
            for i in range(g_len):
                ai = group[i]
                ai_out = ai & ~y
                for j in range(i + 1, g_len):
                    aj = group[j]
                    aj_out = aj & ~y
                    if (ai_out & aj_out) == 0:
                        return False
        return True
    
    # Helper to convert mask to list
    def mask_to_list(mask):
        return [i + 1 for i in range(16) if mask & (1 << i)]
    
    # Main search loop
    # We will run multiple iterations of "Greedy + Swap Local Search"
    
    while time.time() - start_time < time_limit:
        # 1. Greedy Construction
        # Use a different random seed for each iteration to vary the order
        current_seed = int(time.time() * 1000000) % (2**32)
        random.seed(current_seed)
        indices = list(range(n))
        random.shuffle(indices)
        
        chosen = []
        chosen_set = set() # For O(1) lookup
        
        # Greedy Pass
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            c = all_sets[idx]
            if c in chosen_set:
                continue
            if can_add(c, chosen):
                chosen.append(c)
                chosen_set.add(c)
        
        # 2. Local Search (Swaps)
        # Try to improve by swapping
        improved = True
        swap_attempts = 0
        max_swap_attempts = 200 # Limit per iteration to prevent timeout
        
        while improved and swap_attempts < max_swap_attempts:
            improved = False
            swap_attempts += 1
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random set to remove
            if not chosen:
                break
            rem_idx = random.randint(0, len(chosen) - 1)
            rem_mask = chosen[rem_idx]
            
            # Remove it
            chosen.pop(rem_idx)
            chosen_set.remove(rem_mask)
            
            # Try to add a random set
            added = False
            # Try a few random candidates
            for _ in range(5):
                add_idx = random.randint(0, n - 1)
                add_mask = all_sets[add_idx]
                if add_mask in chosen_set:
                    continue
                if can_add(add_mask, chosen):
                    chosen.append(add_mask)
                    chosen_set.add(add_mask)
                    added = True
                    improved = True
                    break
            
            if not added:
                # Re-add the removed set
                chosen.append(rem_mask)
                chosen_set.add(rem_mask)
        
        # Update best
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
        
        # If we are getting close to the theoretical max or time is running out, we can stop
        # But let's keep searching until time limit.
        
    # Convert best_family to list of lists
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
    
    return result
