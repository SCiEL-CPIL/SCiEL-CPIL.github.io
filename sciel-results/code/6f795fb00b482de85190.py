"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Strategy: Greedy selection to avoid subset sum 200
    # We try different orders of consideration to maximize cardinality.
    
    def run_greedy(order):
        selected = []
        reachable = {0}
        # Pre-compute potential sums to speed up check? 
        # reachable size is <= 200.
        for x in order:
            # Check if adding x creates sum 200
            # This happens if 200 - x is in reachable
            if 200 - x in reachable:
                continue
            
            # If safe, add x
            selected.append(x)
            
            # Update reachable sums
            # We only care about sums <= 200
            new_sums = []
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.append(ns)
            
            # Add new sums to reachable
            reachable.update(new_sums)
            
        return selected

    candidates = list(range(1, 101))
    
    best_set = []
    
    # 1. Descending order (Inspiration's approach)
    order_desc = sorted(candidates, reverse=True)
    set_desc = run_greedy(order_desc)
    if len(set_desc) > len(best_set):
        best_set = set_desc
        
    # 2. Ascending order
    order_asc = sorted(candidates)
    set_asc = run_greedy(order_asc)
    if len(set_asc) > len(best_set):
        best_set = set_asc
        
    # 3. Random shuffles to explore different local optima
    # Budget: 50s. Each run is very fast (< 1ms). 
    # We can do many shuffles.
    max_iter = 10000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        set_rand = run_greedy(candidates)
        if len(set_rand) > len(best_set):
            best_set = set_rand
            # Optional: if we find a very large set, we could stop? 
            # But let's keep searching for the max.
            
    return best_set
# EVOLVE-BLOCK-END
