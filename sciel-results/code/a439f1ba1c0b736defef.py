import time
from fractions import Fraction

def construct():
    """
    Returns a nonempty subset of [1, 24] maximizing cardinality
    such that no reciprocal of a selected element equals the sum of 
    reciprocals of a subset of the other selected elements.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s limit

    S = []
    # Set of all possible sums of reciprocals of elements in S
    # Initially, the empty subset sums to 0
    sums = {Fraction(0)}

    # We process elements in decreasing order (24 down to 1).
    # If we add 'a', we must check if 1/a is a sum of reciprocals of 
    # already selected elements (which are all > a).
    # If 1/a is NOT such a sum, we can add 'a'.
    # We do NOT need to check if adding 'a' subsumes existing elements 'b' > a,
    # because 1/a > 1/b, so any sum involving 1/a is > 1/b.
    
    for a in range(24, 0, -1):
        if time.time() > deadline:
            break

        target = Fraction(1, a)
        
        # Check if target is already in the set of achievable sums
        if target in sums:
            # 1/a is formed by a subset of S. 
            # Adding a would make a subsumed.
            continue
        else:
            # 1/a is not formed by any subset of S.
            # Safe to add.
            S.append(a)
            
            # Update the set of achievable sums
            # New sums = old sums union {s + 1/a for s in old sums}
            # We create a new set to avoid modifying while iterating
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + target)
            sums = new_sums

    # The problem requires a nonempty subset.
    # The greedy algorithm will always add at least 24 (since sums={0} initially, 1/24 not in {0}).
    # So S should not be empty.
    if not S:
        return [24]
        
    return S
