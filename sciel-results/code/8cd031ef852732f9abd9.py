import time
import random

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets 
    have different cardinalities, their sums differ. Maximize set size.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_set = list(range(128, 136))  # Fallback to seed if nothing better found quickly
    
    def is_valid_addition(current_sums, x):
        """
        Check if adding x to the set represented by current_sums maintains the invariant.
        current_sums is a list of sets, where current_sums[k] is the set of sums of subsets of size k.
        Returns True if valid, False otherwise.
        """
        n = len(current_sums)
        # We are adding one element, so new size is n+1
        # new_sums[k] = current_sums[k] U {s + x for s in current_sums[k-1]}
        
        # We can check disjointness on the fly or construct the new list and check.
        # Constructing the new list is O(N * |sums|). Checking disjointness is O(N^2 * |sums|).
        # Given N is small (< 60), this is fine.
        
        new_sums = [set() for _ in range(n + 1)]
        
        # Copy old sums
        for k in range(n):
            new_sums[k].update(current_sums[k])
        
        # Add new sums involving x
        for k in range(n + 1):
            if k == 0:
                continue
            if k - 1 < n:
                for s in current_sums[k-1]:
                    new_sums[k].add(s + x)
        
        # Check disjointness between all different cardinalities
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if not new_sums[i].isdisjoint(new_sums[j]):
                    return False
        return True

    def greedy_construct(order):
        """
        Greedily add elements from 'order' to the set if valid.
        """
        current_set = []
        # current_sums[0] = {0}, current_sums[k] = set of sums
        current_sums = [set([0])]
        
        for x in order:
            if is_valid_addition(current_sums, x):
                current_set.append(x)
                # Update current_sums
                n = len(current_sums)
                new_sums = [set() for _ in range(n + 1)]
                for k in range(n):
                    new_sums[k].update(current_sums[k])
                for k in range(n + 1):
                    if k > 0:
                        for s in current_sums[k-1]:
                            new_sums[k].add(s + x)
                current_sums = new_sums
                
        return current_set

    # Strategy 1: Descending order (Large numbers first)
    # This tends to work well because large numbers spread out sums.
    order_desc = list(range(256, 0, -1))
    set_desc = greedy_construct(order_desc)
    if len(set_desc) > len(best_set):
        best_set = set_desc
        
    # Strategy 2: Ascending order
    order_asc = list(range(1, 257))
    set_asc = greedy_construct(order_asc)
    if len(set_asc) > len(best_set):
        best_set = set_asc

    # Strategy 3: Randomized restarts
    # We have time to try a few random permutations.
    # To make it efficient, we can try random shuffles of the full range or subsets.
    # Given the time, let's try a few random permutations of the full range.
    
    # Pre-generate a few random orders
    random.seed(42) # Reproducibility for debugging, but random is fine
    num_restarts = 5
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        order_rand = list(range(1, 257))
        random.shuffle(order_rand)
        set_rand = greedy_construct(order_rand)
        if len(set_rand) > len(best_set):
            best_set = set_rand
            
    # Strategy 4: Local search / Refinement?
    # If we have a good set, can we swap elements to make room for more?
    # This is complex. The greedy approach with good ordering is usually strong.
    # Let's try a "middle-out" or specific heuristic ordering if time permits.
    # Actually, let's try an ordering that prioritizes numbers that are "far" from the current sum centers.
    # But simple random restarts might be sufficient.
    
    # One more specific heuristic: 
    # Sort by (value - 128)^2 to prioritize extremes?
    order_extremes = sorted(range(1, 257), key=lambda x: abs(x - 128))
    # This puts 1, 256, 2, 255... 
    # Or sort by value, but start from middle?
    # Let's try an order that starts with large numbers, then smaller.
    # The descending order already does this.
    
    # Let's try a "greedy with backtracking" for the first few elements?
    # Too slow.
    
    # Let's ensure the result is a list of ints
    return list(best_set)
