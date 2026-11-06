import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin: kill at 90s, leave 3s buffer
    
    N = 100
    target = 200
    mask = (1 << (target + 1)) - 1

    def compute_reachable(S):
        r = 1
        for x in S:
            r = (r | (r << x)) & mask
        return r

    def is_valid(S):
        r = compute_reachable(S)
        return ((r >> target) & 1) == 0

    def is_valid_with_reachable(reachable):
        return ((reachable >> target) & 1) == 0

    # 1. Generate initial candidates using Randomized Greedy with incremental reachability
    best_set = set()
    best_size = 0
    
    # Base set from original code is a good deterministic start
    base_S = set(range(66, 101))
    if is_valid(base_S):
        best_set = base_S
        best_size = len(base_S)
        
    candidates = list(range(1, 101))
    
    # Run several randomized greedy trials with incremental reachability
    random.seed(42)
    num_trials = 200
    for i in range(num_trials):
        if time.time() > deadline - 15: break # Leave time for local search
        
        S_try = set()
        reachable = 1  # Start with sum 0 reachable
        
        # Random order
        order = candidates[:]
        random.shuffle(order)
        
        for x in order:
            # Check if adding x keeps target unreachable using incremental reachability
            new_reachable = (reachable | (reachable << x)) & mask
            if is_valid_with_reachable(new_reachable):
                S_try.add(x)
                reachable = new_reachable
                
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # 2. Local Search Refinement
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Precompute reachable for current set
    current_reachable = compute_reachable(current_set)
    
    no_improvement_count = 0
    max_no_improvement = 100
    
    while time.time() < deadline - 2:
        improved = False
        
        # Phase 1: Try to add as many elements as possible
        # We loop until no more elements can be added in a single pass
        # This is more aggressive than adding one at a time
        while True:
            if time.time() > deadline - 2:
                break
            added_any = False
            not_in_set = [x for x in candidates if x not in current_set]
            for x in not_in_set:
                new_reachable = (current_reachable | (current_reachable << x)) & mask
                if is_valid_with_reachable(new_reachable):
                    current_set.add(x)
                    current_reachable = new_reachable
                    current_size += 1
                    improved = True
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
            if not added_any:
                break
        
        # Phase 2: If no additions were made, try swaps
        if not improved and current_set:
            not_in_set = [x for x in candidates if x not in current_set]
            if not_in_set:
                # Try all possible swaps to escape local optimum
                # We look for a swap that is valid. 
                # To be more effective, we can try to find a swap that allows 
                # the set to grow later, but for now, any valid swap helps explore.
                # Optimization: Iterate u in current_set, v in not_in_set
                swap_found = False
                u_list = list(current_set)
                for u in u_list:
                    if time.time() > deadline - 2:
                        break
                    # Calculate reachable without u? 
                    # Removal is hard. We must recompute reachable for (S - {u}) | {v}
                    # To speed this up, we can precompute reachable for S - {u}?
                    # That's expensive. 
                    # Given N=100, |S|~80, |not_in_set|~20. 
                    # 80*20 = 1600 checks. Each check is O(100) bit ops. 
                    # 160,000 bit ops is very fast in Python if optimized? 
                    # compute_reachable is O(|S|). 
                    # So 1600 * 80 = 128,000 operations. This is fine.
                    
                    for v in not_in_set:
                        new_set = (current_set - {u}) | {v}
                        # Check validity
                        # We can optimize by checking if v is "compatible" with S\u
                        # But S\u is not tracked.
                        # Let's just call is_valid.
                        if is_valid(new_set):
                            current_set = new_set
                            current_reachable = compute_reachable(current_set)
                            improved = True
                            swap_found = True
                            break
                    if swap_found:
                        break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try a fresh greedy from scratch with different seed
                if time.time() > deadline - 10:
                    break
                # Restart with a new random greedy
                S_try = set()
                reachable = 1
                order = candidates[:]
                random.shuffle(order)
                for x in order:
                    new_reachable = (reachable | (reachable << x)) & mask
                    if is_valid_with_reachable(new_reachable):
                        S_try.add(x)
                        reachable = new_reachable
                
                if len(S_try) >= current_size:
                    current_set = S_try
                    current_reachable = compute_reachable(current_set)
                    current_size = len(current_set)
                    no_improvement_count = 0
                else:
                    break
        else:
            no_improvement_count = 0
        
    return list(best_set)
