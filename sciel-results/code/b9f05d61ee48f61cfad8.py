"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce
import random

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin: 55s out of 60s
    
    # Compute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    vals = {a: L // a for a in range(1, 23)}
    
    # Precompute all possible elements
    elements = list(range(1, 23))
    
    def greedy_build(order):
        """Build a dissociated set by greedily adding elements in the given order."""
        current = []
        sums = {0}
        for a in order:
            v = vals[a]
            shifted = {s + v for s in sums}
            if not (shifted & sums):
                current.append(a)
                sums = sums | shifted
        return current, sums

    def try_extend(current, sums):
        """Try to add any remaining element to the current set."""
        current_set = set(current)
        remaining = [a for a in elements if a not in current_set]
        changed = True
        while changed:
            changed = False
            for a in remaining:
                v = vals[a]
                shifted = {s + v for s in sums}
                if not (shifted & sums):
                    current.append(a)
                    sums = sums | shifted
                    current_set.add(a)
                    remaining.remove(a)
                    changed = True
                    break # Restart loop to update remaining and sums
        return current, sums

    best_subset = []
    best_sums = {0}
    
    # Strategy 1: Deterministic Greedies
    order_desc_weight = sorted(elements, key=lambda a: vals[a], reverse=True)
    current, sums = greedy_build(order_desc_weight)
    current, sums = try_extend(current, sums)
    if len(current) > len(best_subset):
        best_subset = current
        best_sums = sums
        
    order_asc_weight = sorted(elements, key=lambda a: vals[a])
    current, sums = greedy_build(order_asc_weight)
    current, sums = try_extend(current, sums)
    if len(current) > len(best_subset):
        best_subset = current
        best_sums = sums
        
    # Strategy 2: Randomized Greedy
    random.seed(42)
    # We can afford many restarts
    max_restarts = 20000 
    for i in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(elements)
        current, sums = greedy_build(elements)
        current, sums = try_extend(current, sums)
        
        if len(current) > len(best_subset):
            best_subset = current
            best_sums = sums
            
    # Strategy 3: Local Search on Best Candidate
    if time.time() < deadline - 10 and len(best_subset) < 22:
        current = list(best_subset)
        sums = best_sums
        current_set = set(current)
        remaining = [a for a in elements if a not in current_set]
        
        # Try swaps to potentially increase size
        # We limit the number of swap attempts to avoid timeout
        swap_attempts = 0
        max_swap_attempts = 1000
        improved = True
        while improved and swap_attempts < max_swap_attempts and time.time() < deadline - 5:
            improved = False
            # Shuffle remaining and current to randomize search
            random.shuffle(remaining)
            random.shuffle(current)
            
            for i in range(len(current)):
                for j in range(len(remaining)):
                    swap_attempts += 1
                    if swap_attempts >= max_swap_attempts:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    old_elem = current[i]
                    new_elem = remaining[j]
                    
                    # Create candidate set
                    # Remove old, add new
                    # Rebuild sums is expensive, so we check validity first?
                    # No, we need sums to check validity.
                    # We can try to calculate the new sums set.
                    # Since we are swapping, the size is same.
                    # If valid, we then try to extend.
                    
                    # Optimization: Only try swap if it might lead to extension?
                    # Hard to know. Just try.
                    
                    test_subset = current[:i] + current[i+1:] + [new_elem]
                    
                    # Fast validity check
                    valid = True
                    test_sums = {0}
                    for a in test_subset:
                        v = vals[a]
                        shifted = {s + v for s in test_sums}
                        if shifted & test_sums:
                            valid = False
                            break
                        test_sums = test_sums | shifted
                    
                    if valid:
                        # Try to extend
                        test_current, test_sums = try_extend(test_subset, test_sums)
                        if len(test_current) > len(current):
                            current = test_current
                            sums = test_sums
                            current_set = set(current)
                            remaining = [a for a in elements if a not in current_set]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if len(current) > len(best_subset):
            best_subset = current
            best_sums = sums

    # Ensure non-empty
    if not best_subset:
        best_subset = [1]
        
    return best_subset

# EVOLVE-BLOCK-END
