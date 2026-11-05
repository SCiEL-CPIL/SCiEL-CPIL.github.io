import sys
from math import gcd
from functools import reduce

def construct():
    # Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23))
    
    # Integer values v_i = L // i
    vals = [L // i for i in range(1, 23)]
    indices = list(range(1, 23))
    
    # Map index to value
    # vals[i-1] is L//i
    
    def check_collision(current_sums, new_val):
        """
        Check if adding new_val to a set with subset sums 'current_sums'
        creates a collision.
        """
        new_sums = set()
        for s in current_sums:
            ns = s + new_val
            if ns in current_sums:
                return True # Collision
            new_sums.add(ns)
        return False

    def construct_greedy(order):
        """
        Construct a dissociated set by iterating through indices in 'order'.
        If adding the index's value doesn't cause a collision, add it.
        """
        current_sums = {0}
        selected = []
        
        for idx in order:
            val = vals[idx - 1]
            if not check_collision(current_sums, val):
                # Safe to add
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + val)
                current_sums.update(new_sums)
                selected.append(idx)
        
        return selected

    # Strategy 1: Descending Value (Largest first)
    # Larger values are more "distinct" from sums of smaller ones?
    # Actually, smaller values sum up to larger values more easily.
    # So picking large ones first might block small ones that sum to them.
    # But picking small ones first might create a dense sum set that blocks large ones.
    # Let's try both.
    
    order_desc = sorted(indices, key=lambda i: vals[i-1], reverse=True)
    set_desc = construct_greedy(order_desc)
    
    # Strategy 2: Ascending Value (Smallest first)
    order_asc = sorted(indices, key=lambda i: vals[i-1])
    set_asc = construct_greedy(order_asc)
    
    # Strategy 3: Original Order (1 to 22)
    set_orig = construct_greedy(indices)
    
    # Strategy 4: Reverse Original Order (22 to 1)
    set_rev = construct_greedy(list(reversed(indices)))
    
    # Strategy 5: Prime Indices First, then Composite
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    composites = [i for i in indices if i not in primes]
    order_prime_first = primes + composites
    set_prime = construct_greedy(order_prime_first)
    
    # Strategy 6: Composite Indices First, then Prime
    order_comp_first = composites + primes
    set_comp = construct_greedy(order_comp_first)
    
    # Select the best (largest cardinality)
    candidates = [set_desc, set_asc, set_orig, set_rev, set_prime, set_comp]
    
    best_set = max(candidates, key=len)
    
    # Fallback to known good solution if something went wrong (shouldn't happen)
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
