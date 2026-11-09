import time

def construct():
    deadline = time.time() + 50  # safety margin
    
    # Start with the set {67, 68, ..., 100} which is known to avoid sum 200
    # Size = 34
    base = set(range(67, 101))
    best = set(base)
    
    # Try to add elements from [1, 66] one by one
    # Check if adding x to current set still avoids subset sum 200
    def has_sum_200(S):
        """Check if any subset of S sums to 200"""
        reachable = {0}
        for x in S:
            new_reachable = set()
            for s in reachable:
                ns = s + x
                if ns == 200:
                    return True
                if ns < 200:
                    new_reachable.add(ns)
            reachable |= new_reachable
        return False
    
    # Greedily try to add each element from 1 to 66
    for x in range(1, 67):
        if time.time() > deadline:
            break
        candidate = best | {x}
        if not has_sum_200(candidate):
            best = candidate
    
    return list(best)
