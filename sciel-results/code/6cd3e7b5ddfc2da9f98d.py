import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    weights = {s: L // s for s in range(1, 23)}

    def is_dissociated(S):
        if not S:
            return True
        w = [weights[s] for s in S]
        sums = {0}
        for val in w:
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    best_set = []
    best_size = 0

    # Strategy 1: Greedy with random restarts
    # Use fewer restarts to leave time for local search
    for _ in range(50):
        if time.time() > deadline - 10:  # Reserve 10s for local search
            break
        
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        for x in cand_order:
            new_set = current_set + [x]
            if is_dissociated(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Strategy 2: Local Search (Swap/Add)
    # Start from best_set
    current_best = list(best_set)
    
    # Try to improve
    # We can also try to start local search from other good candidates found during greedy
    # But best_set is the best so far.
    
    # To diversify, we can also keep a few other good sets.
    # For simplicity, just stick with best_set.
    
    while time.time() < deadline - 2:
        if not current_best:
            break
            
        improved = False
        
        # Try to add a random missing element
        missing = [x for x in range(1, 23) if x not in current_best]
        if missing:
            x = random.choice(missing)
            new_set = current_best + [x]
            if is_dissociated(new_set):
                current_best = new_set
                improved = True
        
        # Try to swap: remove one, add another
        if not improved and len(current_best) > 1 and missing:
            x_remove = random.choice(current_best)
            x_add = random.choice(missing)
            if x_remove != x_add:
                new_set = [x for x in current_best if x != x_remove] + [x_add]
                if is_dissociated(new_set):
                    current_best = new_set
                    improved = True
        
        # If no improvement, try a "kick": remove 2, add 2?
        # Or just continue.
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = current_best

    if not best_set:
        return [1]
        
    return best_set
