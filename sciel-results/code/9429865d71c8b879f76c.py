import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
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

    # 1. Generate initial candidates using Randomized Greedy
    best_set = set()
    best_size = 0
    
    # Base set from original code is a good deterministic start
    base_S = set(range(66, 101))
    if is_valid(base_S):
        best_set = base_S
        best_size = len(base_S)
        
    candidates = list(range(1, 101))
    
    # Run several randomized greedy trials
    random.seed(42)
    for i in range(100): # Do more trials
        if time.time() > deadline - 5: break # Leave time for local search
        
        S_try = set()
        # Random order
        order = candidates[:]
        random.shuffle(order)
        
        for x in order:
            # Check if adding x is valid
            # We can maintain reachable incrementally for speed
            # But for simplicity in this sketch, let's just check is_valid(S_try U {x})
            # This is O(N^2) per trial. 100 trials * 100 elements * 100 ops = 1e6 ops. Fine.
            new_S = S_try | {x}
            if is_valid(new_S):
                S_try = new_S
                
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # 2. Local Search Refinement
    # We will try to improve best_set
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Precompute reachable for current set
    current_reachable = compute_reachable(current_set)
    
    no_improvement_count = 0
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try to add an element
        # Pick a random element not in set
        not_in_set = [x for x in candidates if x not in current_set]
        if not_in_set:
            x = random.choice(not_in_set)
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
        
        # Try to swap
        # Pick u in set, v not in set
        if current_set and not_in_set:
            u = random.choice(list(current_set))
            v = random.choice(not_in_set)
            if u != v:
                new_set = (current_set - {u}) | {v}
                if is_valid(new_set):
                    # Swap doesn't change size, but might enable future adds
                    # We accept swap if it leads to a "better" state?
                    # Since size is same, we accept it to explore.
                    # But we must ensure we don't cycle.
                    # Simple acceptance: always accept valid swap.
                    current_set = new_set
                    improved = True
        
        if not improved:
            no_improvement_count += 1
            # If stuck, try a random restart from a different greedy solution?
            # Or just break and try another portfolio member?
            if no_improvement_count > 1000:
                break
        else:
            no_improvement_count = 0
            
        # Update reachable if we changed set?
        # is_valid recomputes it. For efficiency, we could update incrementally,
        # but is_valid is fast enough.
        
    return list(best_set)
