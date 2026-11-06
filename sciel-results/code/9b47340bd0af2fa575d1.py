import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))

    # Convert to integer weights
    # weight for s is L // s
    weights = {s: L // s for s in range(1, 23)}

    def is_dissociated_integer(S):
        """
        Checks if S (list of ints from 1..22) is dissociated.
        Uses integer arithmetic with LCM.
        """
        if not S:
            return True

        # Get the weights for the elements in S
        w = [weights[s] for s in S]

        # Generate all subset sums
        sums = [0]
        for val in w:
            # Add val to all existing sums
            sums += [s + val for s in sums]

        # Check for duplicates
        # Sorting is efficient
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums) - 1):
            if sorted_sums[i] == sorted_sums[i+1]:
                return False
        return True

    def is_dissociated_fast(S):
        """
        Optimized check using set for early termination.
        """
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

    # Strategy 1: Increasing order
    current_set = []
    for x in range(1, 23):
        if time.time() > deadline: break
        new_set = current_set + [x]
        if is_dissociated_fast(new_set):
            current_set = new_set

    if len(current_set) > len(best_set):
        best_set = current_set

    # Strategy 2: Decreasing order
    current_set = []
    for x in range(22, 0, -1):
        if time.time() > deadline: break
        new_set = current_set + [x]
        if is_dissociated_fast(new_set):
            current_set = new_set

    if len(current_set) > len(best_set):
        best_set = current_set

    # Strategy 3: Randomized Greedy with multiple restarts
    # We want to maximize cardinality.
    # The known best is around 11-12.
    # Let's try to find larger sets.
    
    # To improve on greedy, we can use a "repair" or "swap" strategy,
    # but simple randomized greedy with different seeds often helps.
    
    # Also, consider that some numbers are "harder" to include.
    # Numbers with large weights (small s) are more likely to cause collisions?
    # Actually, large weights spread sums out.
    # Small weights (large s) are closer to 0, might be easier to pack?
    
    # Let's try prioritizing larger numbers first in random order too.
    
    for _ in range(200):
        if time.time() > deadline: break
        
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        for x in cand_order:
            new_set = current_set + [x]
            if is_dissociated_fast(new_set):
                current_set = new_set
                
        if len(current_set) > len(best_set):
            best_set = current_set

    # Strategy 4: Local Search / Swap
    # Start from best_set found so far.
    # Try to swap elements or add/remove to increase size.
    
    current_best = list(best_set)
    
    # Try to add any missing element
    missing = [x for x in range(1, 23) if x not in current_best]
    
    # Try adding each missing element
    for x in missing:
        if time.time() > deadline: break
        new_set = current_best + [x]
        if is_dissociated_fast(new_set):
            current_best = new_set
            missing.remove(x)
            # If we added one, we might be able to add more, so restart this loop
            missing = [x for x in range(1, 23) if x not in current_best]
            break
    
    if len(current_best) > len(best_set):
        best_set = current_best

    # Try swapping: remove one element, add another
    # This is more complex. Let's try a simple version.
    # Iterate over all pairs (i in set, j not in set)
    # If swapping i for j results in a dissociated set, do it.
    # This doesn't change size, so it's only useful if it allows adding more later.
    # Given time constraints, let's stick to simple additions and randomized greedy.
    
    # Another strategy: Start with a random subset of size k and try to grow it.
    # Or start with a random subset and try to optimize.
    
    # Let's try a few more randomized greedy passes with different biases.
    # Bias towards larger numbers?
    for _ in range(100):
        if time.time() > deadline: break
        
        # Bias: larger numbers first
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        # Sort with some randomness, favoring larger numbers
        # Actually, just shuffling is fine.
        
        current_set = []
        for x in cand_order:
            new_set = current_set + [x]
            if is_dissociated_fast(new_set):
                current_set = new_set
                
        if len(current_set) > len(best_set):
            best_set = current_set

    # If we found a very good set, return it
    # Fallback if nothing found (shouldn't happen)
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
