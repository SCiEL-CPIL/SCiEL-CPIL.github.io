"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    
    # Configuration: 1 set of size 0, 12 sets of size 1, 15 sets of size 10, 12 sets of size 11
    # Total: 1 + 12 + 15 + 12 = 40
    
    sets_list = []
    
    # Size 0
    sets_list.append([])
    
    # Size 1: All 12 singletons
    for i in range(n):
        sets_list.append([i])
        
    # Size 11: All 12 sets of size 11 (complements of singletons)
    for i in range(n):
        s = [j for j in range(n) if j != i]
        sets_list.append(s)
        
    # Size 10: Pick 15 sets of size 10
    # To maximize comparability with Size 1, we just need any 15.
    # To maximize comparability with Size 11, we want them to be subsets of the Size 11 sets.
    # Every size 10 set is a subset of exactly 2 size 11 sets.
    # Since we have ALL size 11 sets, every size 10 set we pick will be comparable to 2 size 11 sets.
    # So the choice of which 15 size 10 sets doesn't affect P(10,11) (always 2 per set).
    # It doesn't affect P(0,10) (always 1 per set).
    # It affects P(1,10).
    # Each size 1 set is contained in 55 size 10 sets.
    # We pick 15 of 66.
    # Random selection is sufficient for the expected value, but we can just take the first 15.
    
    size_10_sets = []
    for combo in itertools.combinations(range(n), 10):
        size_10_sets.append(list(combo))
        if len(size_10_sets) == 15:
            break
            
    sets_list.extend(size_10_sets)
    
    # Ensure distinctness and correctness
    # Convert to tuples for hashing to check distinctness
    unique_sets = []
    seen = set()
    for s in sets_list:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            unique_sets.append(list(s))
            
    # If we have more than 40, trim. If less, add more.
    # In this construction, we have exactly 40.
    
    return unique_sets[:40]

# EVOLVE-BLOCK-END
