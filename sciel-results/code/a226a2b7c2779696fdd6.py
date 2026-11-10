"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    candidates = list(range(1, 23))
    reciprocals = {x: Fraction(1, x) for x in candidates}
    
    best_set = []
    best_size = 0
    
    def backtrack(order, idx, current_sums, current_set, current_size):
        """
        Recursively tries to add elements from order starting at idx.
        """
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning: if even adding all remaining can't beat best, stop
        if current_size + (len(order) - idx) <= best_size:
            return
            
        if idx == len(order):
            if current_size > best_size:
                best_set = current_set[:]
                best_size = current_size
            return

        elem = order[idx]
        r = reciprocals[elem]
        
        # Option 1: Include elem
        # Check if r is a difference of two existing sums
        # Equivalent to: (current_sums + r) intersects current_sums
        # We do this by generating new sums and checking
        new_sums = set()
        conflict = False
        # Optimization: iterate over list to avoid changing set size issues, though we use a new set
        for s in current_sums:
            ns = s + r
            if ns in current_sums:
                conflict = True
                break
            new_sums.add(ns)
            
        if not conflict:
            # Combine
            new_sums.update(current_sums)
            backtrack(order, idx + 1, new_sums, current_set + [elem], current_size + 1)
            
        # Option 2: Exclude elem
        backtrack(order, idx + 1, current_sums, current_set, current_size)

    # Orderings
    # 1. Descending denominator (Small reciprocals first) - Best heuristic
    order1 = sorted(candidates, reverse=True)
    
    # 2. Ascending denominator (Large reciprocals first)
    order2 = sorted(candidates)
    
    # 3. Interleaved
    order3 = []
    for i in range(11):
        order3.append(22 - i) # Large denom
        order3.append(i + 1)  # Small denom
        
    # 4. Random
    rng = random.Random(42)
    orders_to_try = [order1, order2, order3]
    for _ in range(5):
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders_to_try.append(rand_order)

    for order in orders_to_try:
        if time.time() > deadline:
            break
        initial_sums = {Fraction(0, 1)}
        backtrack(order, 0, initial_sums, [], 0)

    if not best_set:
        best_set = [1, 2, 4, 8, 16]

    return best_set
# EVOLVE-BLOCK-END
