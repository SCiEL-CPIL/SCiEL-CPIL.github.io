import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    # Initial set: all numbers from 129 to 256 form a clique
    S = set(range(129, 257))
    
    # Precompute conflicts for all pairs? No, calculate on the fly.
    # To speed up, we can precompute the "incompatible" set for each x in [2, 256]
    # incompatible[x] = { y in [2, 256] | lcm(x, y) <= 256 }
    
    # However, we only care about y in S.
    
    # Let's use a greedy approach with local search.
    # We try to add/swap in numbers from 2 to 128.
    
    # To allow for more complex swaps, we can iterate multiple times.
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Try candidates in increasing order of value (decreasing reciprocal)
        # Actually, we should try all candidates and pick the best improvement?
        # Or just iterate?
        # Let's iterate through candidates 2..128 and apply if beneficial.
        
        # We need to be careful: modifying S while iterating candidates.
        # It's better to collect all potential swaps and apply the best one,
        # or just apply them sequentially if they don't interfere.
        # Since removing elements from S only makes it easier for others to be added,
        # sequential application is safe and monotonic in terms of feasibility.
        # But it might not be optimal.
        
        # Let's try a simpler loop:
        for x in range(2, 129):
            if x in S:
                continue
            
            # Find conflicts in S
            conflicts = []
            for y in S:
                if lcm(x, y) <= 256:
                    conflicts.append(y)
            
            if not conflicts:
                # Can add directly
                S.add(x)
                improved = True
            else:
                # Check if swap is beneficial
                # Sum of reciprocals of conflicts
                sum_conflicts = sum(1.0 / y for y in conflicts)
                if 1.0 / x > sum_conflicts:
                    # Perform swap
                    for y in conflicts:
                        S.remove(y)
                    S.add(x)
                    improved = True
    
    # After greedy pass, we might have missed some optimizations where
    # removing a set of elements allows adding multiple new ones.
    # Given the time constraint, the greedy pass is likely sufficient for a big jump.
    # The initial set is already very good.
    
    return list(S)
