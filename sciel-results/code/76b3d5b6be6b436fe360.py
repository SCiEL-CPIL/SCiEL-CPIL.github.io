"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _compute_lcm_range(n):
    l = 1
    for i in range(1, n + 1):
        l = _lcm(l, i)
    return l

# Precompute LCM for 1..22
_LCM_22 = _compute_lcm_range(22)

def construct():
    """
    Find a nonempty subset of [1, 22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum = 0) are distinct.
    
    Uses integer arithmetic with a global LCM to speed up checks.
    """
    candidates = list(range(1, 23))
    # Precompute integer values corresponding to 1/s
    # sum(1/s_i) = (1/LCM) * sum(LCM/s_i)
    # Distinctness of rational sums is equivalent to distinctness of integer sums
    values = {s: _LCM_22 // s for s in candidates}
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Descending order (largest s, smallest value)
    # Strategy 2: Ascending order (smallest s, largest value)
    # Strategy 3: Many random orderings
    
    # Helper to run greedy for a given order
    def run_greedy(order):
        current = []
        current_sums = {0}  # Integer sums
        
        for candidate in order:
            v = values[candidate]
            new_sums = set()
            # Optimization: if len(current_sums) is large, this is the bottleneck
            for existing in current_sums:
                new_sums.add(existing)
                new_sums.add(existing + v)
            
            if len(new_sums) == 2 * len(current_sums):
                current.append(candidate)
                current_sums = new_sums
            else:
                # If we fail, we just skip this candidate in this greedy pass
                # We do NOT backtrack in this simple greedy
                pass
        
        return current

    # Run fixed orderings
    order_desc = sorted(candidates, reverse=True)
    order_asc = sorted(candidates)
    
    results = []
    results.append(run_greedy(order_desc))
    results.append(run_greedy(order_asc))
    
    # Run many random orderings
    rng = random.Random(12345)
    for _ in range(500):
        order = candidates[:]
        rng.shuffle(order)
        results.append(run_greedy(order))
        
    # Also try some heuristic orderings
    # Sort by value (which is inverse of s)
    order_by_val = sorted(candidates, key=lambda x: values[x])
    results.append(run_greedy(order_by_val))
    results.append(run_greedy(sorted(candidates, key=lambda x: values[x], reverse=True)))
    
    # Find the best result
    for res in results:
        if len(res) > best_size:
            best_size = len(res)
            best_set = res[:]
            
    if not best_set:
        best_set = [1]
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
