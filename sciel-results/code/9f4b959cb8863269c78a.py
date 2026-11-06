import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    # Fixed elements
    initial_set = [1, 2, 5, 11]
    
    # Calculate initial sums
    initial_sums = set()
    for i, x in enumerate(initial_set):
        for y in initial_set[i:]:
            initial_sums.add(x + y)
            
    # Bitset for sums (max sum 568)
    sums_mask = 0
    for s in initial_sums:
        sums_mask |= (1 << s)
        
    best_set = initial_set[:]
    best_size = len(initial_set)
    
    # Candidates
    candidates = list(range(12, 285))
    
    def is_compatible(x, current_set, sums_mask):
        # Check 2*x
        if (sums_mask >> (2 * x)) & 1:
            return False
        # Check x + y
        for y in current_set:
            s = x + y
            if (sums_mask >> s) & 1:
                return False
        return True

    def update_best(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)

    def search(current_set, sums_mask, candidates):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        update_best(current_set)
        
        # Pruning: if even adding all remaining candidates doesn't beat best, stop
        if len(current_set) + len(candidates) <= best_size:
            return
            
        # Heuristic: Try candidates in increasing order
        # We can also try to prioritize candidates that are 'far' from existing ones, 
        # but increasing order is a good baseline for Golomb rulers.
        
        for i, x in enumerate(candidates):
            if time.time() > deadline:
                return
                
            if is_compatible(x, current_set, sums_mask):
                # Add x
                new_sums_mask = sums_mask
                new_sums_mask |= (1 << (2 * x))
                for y in current_set:
                    new_sums_mask |= (1 << (x + y))
                    
                new_set = current_set + [x]
                new_candidates = candidates[i+1:]
                
                search(new_set, new_sums_mask, new_candidates)
                
    search(initial_set, sums_mask, candidates)
    
    return best_set
