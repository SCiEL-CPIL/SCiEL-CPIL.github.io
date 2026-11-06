import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Compute LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23))
    
    # Map i to its scaled value L/i
    # We want to pick a subset of indices {1, ..., 22}
    # The values are V = [L//1, L//2, ..., L//22]
    values = [L // i for i in range(1, 23)]
    
    # 2. Fast Dissociation Check
    def is_dissociated(indices):
        """
        indices: list of indices into 'values'
        Returns True if all subset sums of {values[i] for i in indices} are distinct.
        """
        if not indices:
            return True
            
        # Generate subset sums
        # Start with {0}
        sums = [0]
        for idx in indices:
            v = values[idx]
            # Add v to all existing sums
            # We can do this in place or create new list
            # To avoid modifying while iterating, create new list
            new_sums = [s + v for s in sums]
            sums.extend(new_sums)
            
        # Check for duplicates
        if len(sums) != len(set(sums)):
            return False
        return True

    # 3. Search Strategy
    # We want to maximize the number of elements.
    # Universe is 0..21 (indices for 1..22)
    universe = list(range(22))
    
    best_set = []
    best_size = 0
    
    # Heuristic: Larger indices (smaller values 1/i) are "safer" to add?
    # Or smaller indices (larger values)?
    # Let's try both orders in the greedy step.
    
    def greedy_construct(order):
        current_set = []
        # order is a permutation of universe
        for idx in order:
            if time.time() > deadline:
                break
            new_set = current_set + [idx]
            if is_dissociated(new_set):
                current_set = new_set
        return current_set

    # Try a few fixed orders
    # Order 1: Ascending index (1, 2, ..., 22) -> Values: Large to Small
    # Order 2: Descending index (22, 21, ..., 1) -> Values: Small to Large
    # Order 3: Random
    
    orders = [
        list(range(22)),
        list(range(21, -1, -1)),
        # Maybe sort by value? 
        # values are L/i. L/1 is largest, L/22 is smallest.
        # So range(22) is Large->Small values.
        # range(21,-1,-1) is Small->Large values.
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Randomized Search
    random.seed(42)
    
    # Number of restarts depends on time
    # is_dissociated is O(2^k * k). For k=12, 4096*12 ~ 50k ops. Very fast.
    # We can do many restarts.
    
    for i in range(2000):
        if time.time() > deadline:
            break
        
        # Random permutation
        perm = universe.copy()
        random.shuffle(perm)
        
        s = greedy_construct(perm)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        
        # Local Search / Polish
        # Try to add any remaining element that fits
        # This is already done by greedy if the order allows it, 
        # but greedy might have skipped an element early on that could fit now?
        # No, if it didn't fit then, adding more elements makes the set of sums larger,
        # so if it didn't fit before, it won't fit now?
        # Wait. If s1+v collides, adding another element w creates more sums.
        # It's possible that s1+v was a collision, but with w added, 
        # maybe the "conflict" is resolved? No, the collision is a specific equality.
        # If sum(A) + v == sum(B), this equality holds regardless of other elements.
        # So if v didn't fit, it will never fit in a superset.
        # Therefore, greedy is "maximal" in the sense that no single element 
        # from the remaining can be added to the current set.
        # However, it might not be "maximum" (cardinality).
        # A different set of elements might allow a larger cardinality.
        
        # To find larger cardinality, we might need to swap elements.
        # Let's try a simple swap heuristic on the best set found so far.
        # But doing this in the loop might be expensive.
        # Let's just rely on many random restarts.
        # The space of subsets is 2^22 ~ 4M. 
        # We can't search all. 
        # But random greedy is quite good for dissociated sets.

    # One more polish: 
    # If we found a set of size K, try to see if we can replace one element 
    # to allow adding two more?
    # This is complex. Let's stick to the robust randomized greedy.
    
    # Convert indices back to values 1..22
    # indices are 0..21 corresponding to 1..22
    # idx 0 -> 1
    # idx 1 -> 2
    # ...
    # idx 21 -> 22
    result = [idx + 1 for idx in best_set]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    print(construct())
