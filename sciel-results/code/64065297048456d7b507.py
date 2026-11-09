# EVOLVE-BLOCK-START
import random

def construct():
    def greedy_solve(order):
        selected = []
        reachable = {0}
        for x in order:
            target = 200 - x
            if target in reachable:
                continue
            selected.append(x)
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            reachable |= new_sums
        return selected

    best_set = []
    best_size = 0
    
    # Try descending order first (baseline)
    desc_order = list(range(100, 0, -1))
    s = greedy_solve(desc_order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # Try randomized perturbations of descending order
    # This helps escape local optima of the greedy choice
    for _ in range(20):
        # Create a list of numbers 1-100
        nums = list(range(1, 101))
        # Sort descending, but introduce some randomness
        # We want to keep the general "large first" trend
        # Add small noise to the sort key
        noisy_nums = [(x, random.uniform(0, 10)) for x in nums]
        noisy_nums.sort(key=lambda t: -t[0] + t[1]) # -x + noise
        order = [x for x, _ in noisy_nums]
        
        s = greedy_solve(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    return best_set
# EVOLVE-BLOCK-END
