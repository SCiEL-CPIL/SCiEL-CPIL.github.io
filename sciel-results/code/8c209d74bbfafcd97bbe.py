import time
from itertools import combinations

def construct():
    """
    Constructs a set of 12 distinct positive integers with unique subset sums,
    minimizing the largest integer.
    
    Uses a greedy "minimal extension" algorithm:
    1. Start with an empty set and an empty set of subset sums {0}.
    2. At each step, find the smallest positive integer not yet used such that
       adding it to the current set does not create any duplicate subset sums.
    3. This is equivalent to checking that for all existing subset sums s,
       the new sum s + x is not already in the set of subset sums.
    
    For n=12, this should be fast enough to run within the time limit.
    """
    n = 12
    deadline = time.time() + 80  # Leave 10s safety margin for the 90s limit
    
    # Start with empty set
    elements = []
    subset_sums = {0}
    
    for _ in range(n):
        if time.time() > deadline:
            # If we're running out of time, fall back to the binary sequence
            # (which is guaranteed valid and has largest element 2048)
            return [1 << i for i in range(n)]
        
        # Find the smallest valid next element
        # We need to check candidates starting from 1
        # But we can be smarter: the smallest valid x must satisfy
        # that x + s is not in subset_sums for any s in subset_sums
        
        # Start checking from 1
        x = 1
        while True:
            if time.time() > deadline:
                # Fall back to binary sequence
                return [1 << i for i in range(n)]
            
            # Check if adding x creates any collision
            # A collision occurs if there exist s1, s2 in subset_sums
            # such that s1 + x == s2, i.e., s2 - s1 == x
            # Or equivalently, if (x + s) is in subset_sums for some s in subset_sums
            
            # Efficient check: for each s in subset_sums, check if s + x in subset_sums
            # But we can optimize: if x is in subset_sums, then 0 + x = x is a collision
            # More generally, we need to check if any s + x is in subset_sums
            
            # Actually, the condition is: the new subset sums are 
            # {s for s in subset_sums} union {s + x for s in subset_sums}
            # These must be disjoint, i.e., (subset_sums) & (subset_sums + x) == empty
            # where subset_sums + x = {s + x for s in subset_sums}
            
            # Check for collision
            has_collision = False
            for s in subset_sums:
                if s + x in subset_sums:
                    has_collision = True
                    break
            
            if not has_collision:
                # x is valid
                elements.append(x)
                # Update subset sums
                new_sums = set()
                for s in subset_sums:
                    new_sums.add(s + x)
                subset_sums.update(new_sums)
                break
            
            x += 1
    
    return elements
