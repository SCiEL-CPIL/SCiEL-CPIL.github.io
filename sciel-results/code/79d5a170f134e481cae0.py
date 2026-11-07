import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0
    
    def get_subset_sums(elements):
        sums = {0}
        for e in elements:
            sums |= {s + e for s in sums}
        sums.discard(0)
        return sums

    def is_valid(sums):
        if not sums: return True
        s_list = sorted(sums)
        max_s = s_list[-1]
        for a in s_list:
            mult = 2 * a
            while mult <= max_s:
                if mult in sums:
                    return False
                mult += a
        return True

    def check_valid(elements):
        return is_valid(get_subset_sums(elements))

    # Initial candidates: Focus on high values
    # Smaller values create small sums which are highly composite/divisible
    candidates = list(range(256, 1, -1))
    
    best_set = []
    best_size = 0

    # Phase 1: Greedy Construction with Randomization
    # Try to build a set by adding elements one by one.
    # We do this many times with different random orders.
    
    for attempt in range(200):
        if time.time() > deadline: break
        
        # Randomize order, but bias towards larger numbers?
        # Just shuffle.
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current = []
        current_sums = set()
        
        # Greedy Add
        for c in shuffled:
            if time.time() > deadline: break
            
            # Try adding c
            # Optimization: If current size is large, this check is expensive.
            # We can skip if we've already exceeded best_size and time is tight?
            # No, we want to maximize.
            
            # Incremental sum update
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + c)
            new_sums.add(c)
            
            if is_valid(new_sums):
                current.append(c)
                current_sums = new_sums
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
        
        # Phase 2: Local Search / Improvement on the best found so far
        # Or on the current one if it's good.
        # Let's improve the global best.
        
    # Phase 3: Intensive Search on Best Set
    # Try to expand best_set using "1-for-2" swaps.
    if best_set:
        current = best_set[:]
        current_sums = get_subset_sums(current)
        
        # Try to add elements
        added = True
        while added and time.time() < deadline:
            added = False
            # Try adding any remaining element
            for c in candidates:
                if c in current: continue
                
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + c)
                new_sums.add(c)
                
                if is_valid(new_sums):
                    current.append(c)
                    current_sums = new_sums
                    best_size = len(current)
                    best_set = current[:]
                    added = True
                    break # Restart loop with new sum set
        
        # Try 1-for-2 swaps
        # Remove u, add v, w
        # This is expensive. Only do if time permits.
        if time.time() < deadline - 5.0:
             # ... implementation of 1-for-2 ...
             pass

    return best_set
