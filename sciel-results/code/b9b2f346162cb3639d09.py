import time
import numpy as np
from itertools import combinations
import random
import math

# EVOLVE-BLOCK-START
def _precompute_weights():
    # LCM of 1..22
    L = 1
    for i in range(1, 23):
        L = math.lcm(L, i)
    weights = {}
    for i in range(1, 23):
        weights[i] = L // i
    return weights

_GLOBAL_WEIGHTS = _precompute_weights()

def _is_dissociated_subset(S):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    Uses pre-computed global weights for speed.
    """
    if not S:
        return True
    
    # Get the weights for the elements in S
    weights = [_GLOBAL_WEIGHTS[s] for s in S]
    
    # Generate all subset sums
    # Optimization: If n is large, this is the bottleneck.
    # We can use a set to check for duplicates on the fly.
    seen = set()
    seen.add(0)
    
    current_sums = [0]
    for w in weights:
        new_sums = [s + w for s in current_sums]
        for ns in new_sums:
            if ns in seen:
                return False
            seen.add(ns)
        current_sums.extend(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    best_set = []
    best_size = 0
    
    # Strategy:
    # 1. Try several randomized greedy constructions.
    # 2. Use a local search to try to expand the best found set.
    
    # Helper to check if we can add an element to a set quickly
    # We can optimize by checking if the new element's weight causes a collision
    # with existing sums.
    # But _is_dissociated_subset is already fast enough for n < 15.
    
    def try_greedy(elements_order):
        current_set = set()
        for e in elements_order:
            if e in current_set:
                continue
            new_set = current_set | {e}
            if _is_dissociated_subset(new_set):
                current_set = new_set
        return current_set

    # Run multiple random restarts
    # We can also try sorted orders (ascending, descending) as they might be better.
    
    orders = []
    # Random orders
    for _ in range(50):
        elements = list(range(1, 23))
        random.shuffle(elements)
        orders.append(elements)
    
    # Heuristic orders
    orders.append(list(range(1, 23))) # Ascending
    orders.append(list(range(22, 0, -1))) # Descending
    # Descending might be better because larger numbers have smaller reciprocals,
    # leading to smaller weights, which might pack better? Or smaller numbers
    # have larger weights, which are more distinct?
    # Actually, larger weights are "more distinct" in a linear sense?
    # Let's try both.
    
    for order in orders:
        if time.time() > deadline:
            break
        current_set = try_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # If we found a large set, try to expand it further with local search
        if len(current_set) >= 10:
            # Local search: try to swap elements or add more
            # This is expensive, so do it sparingly
            pass
            
    # Local search on the best set found
    if best_set:
        current_set = set(best_set)
        # Try to add any remaining element
        for e in range(1, 23):
            if e not in current_set:
                new_set = current_set | {e}
                if _is_dissociated_subset(new_set):
                    current_set = new_set
                    best_size = len(current_set)
                    best_set = list(current_set)
                    
        # Try swaps: remove one, add two? Or remove one, add one different?
        # Swap: remove x, add y.
        # If we can maintain size, we might unlock a path to a larger set later.
        # But we want to maximize size, so we should prioritize additions.
        # If we can't add, we can try to swap to a "better" configuration that allows addition.
        # This is complex. Let's stick to additions and random restarts.
        
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
