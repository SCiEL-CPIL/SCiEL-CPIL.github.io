"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from fractions import Fraction

def _check_dissociated_incremental(subset):
    """
    Check if all reciprocal subset sums are distinct.
    Uses incremental generation of sums to allow early termination.
    """
    if not subset:
        return True
        
    # Start with the empty sum 0
    current_sums = {Fraction(0)}
    
    for x in subset:
        new_sums = set()
        inv_x = Fraction(1, x)
        # For each existing sum, generate a new sum with x included
        # Also keep the old sums (where x is not included)
        # We need to check if any new sum collides with an old sum or another new sum.
        
        # Optimization: 
        # The new set of sums will be {s, s + inv_x for s in current_sums}
        # We can check for collisions during construction.
        
        # To be efficient, we can just compute the full new set and check size.
        # But checking size of 2^k set is O(2^k).
        # Is there a faster check?
        # If we maintain a sorted list, we can merge? No, Fractions are not easily sortable in a way that helps collision detection without full comparison.
        
        # Let's just generate the new set and check if len(new_set) == 2 * len(current_sums).
        # This is O(2^k) per step. Total O(N * 2^N).
        # For N=12, 12 * 4096 = ~50k ops. Very fast.
        
        next_sums = set()
        for s in current_sums:
            next_sums.add(s)
            next_sums.add(s + inv_x)
            
        if len(next_sums) != 2 * len(current_sums):
            return False
            
        current_sums = next_sums
        
    return True

def construct():
    # We want to maximize cardinality.
    # Max possible is 22, but practically much lower.
    # Let's try to find the maximum K such that a dissociated set of size K exists in [1, 22].
    
    # We'll try K from 12 down to 1.
    # For each K, we'll use a randomized local search.
    
    random.seed(42)
    
    # Pre-compute reciprocals for speed? Not strictly necessary, Fraction is cached.
    
    # Define a heuristic for initial sets:
    # 1. Random
    # 2. Greedy: pick largest available numbers that don't cause immediate conflict?
    #    Actually, larger numbers have smaller reciprocals.
    #    Smaller numbers have larger reciprocals.
    #    Conflicts are more likely with small numbers (1, 2, 3...).
    #    So a good strategy might be to include 1, and then pick large numbers.
    
    def try_find_set_of_size(k, num_restarts=50):
        for _ in range(num_restarts):
            # Initialize a random set of size k
            # To improve success rate, bias towards certain structures?
            # Let's just use random for now, but maybe ensure 1 is included in some trials?
            # Actually, 1 is very "strong" (1/1 = 1). It's hard to collide with 1 unless you have 1/2+1/2 (impossible) or 1/3+... 
            # Wait, 1/1 = 1. Can we make 1 from other reciprocals? 
            # 1/2 + 1/3 + 1/6 = 1. So {1, 2, 3, 6} is NOT dissociated.
            # So including 1 restricts the rest of the set significantly.
            # Maybe it's better to NOT include 1?
            # If we don't include 1, the max sum is smaller.
            
            # Let's try two strategies:
            # A: Include 1.
            # B: Exclude 1.
            
            if _ % 2 == 0:
                # Include 1
                if k == 1:
                    cand = [1]
                else:
                    # Pick k-1 from [2, 22]
                    cand = [1] + random.sample(range(2, 23), k-1)
            else:
                # Exclude 1
                cand = random.sample(range(2, 23), k)
            
            # Local Search: try to fix the set by swapping elements
            # We check if it's valid. If not, we try to swap an element that is part of a collision?
            # Finding the collision is hard. 
            # Alternative: Just try random swaps and check if the new set is valid.
            # If we start with a random set, it's likely invalid.
            # We need a smarter initialization or a better search.
            
            # Let's use a "Greedy Construction" as the initializer instead of pure random.
            # Start with empty set.
            # Try to add elements one by one.
            # Order of candidates: 
            #   If we want to maximize size, we should add elements that are "least likely to collide".
            #   Elements with large values (small reciprocals) are safer?
            #   Or elements that are "independent" in some sense?
            
            # Let's restart the inner loop with a Greedy approach.
            
            # Greedy Attempt
            current_set = []
            # Try adding elements in a specific order.
            # Order: 1, 2, 3, ... 22? Or 22, 21, ... 1?
            # Let's try multiple orders.
            
            # For this restart, let's use a randomized greedy order.
            candidates = list(range(1, 23))
            random.shuffle(candidates)
            
            for x in candidates:
                if len(current_set) >= k:
                    break
                test_set = current_set + [x]
                # Only check if adding x keeps it dissociated
                # We can use the incremental check, but we need to check the whole set.
                # Since k is small (<=12), _check_dissociated_incremental is fast.
                if _check_dissociated_incremental(test_set):
                    current_set.append(x)
            
            if len(current_set) == k:
                return current_set
            
            # If greedy failed, try local search on the best greedy result found so far?
            # Or just fail this restart.
            
        return None

    # Try K from 12 down to 1
    for k in range(12, 0, -1):
        # Increase restarts for smaller K if needed, but 12 should be easy if it exists.
        # Let's do 100 restarts for high K, 50 for lower.
        result = try_find_set_of_size(k, num_restarts=100)
        if result:
            return result
            
    # Fallback: return a known small set
    return [1]

# EVOLVE-BLOCK-END
