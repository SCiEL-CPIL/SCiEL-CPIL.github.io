import time
from math import gcd
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Precompute bad triples and conflicts
    # conflicts[x] = list of pairs (a, b) such that {a, b, x} is a bad triple
    # a, b < x is not required, but we can just store all pairs.
    # To check validity of adding x to S, we check if any (a,b) in conflicts[x] is in S.
    conflicts = {x: [] for x in elements}
    
    # Identify all bad triples
    # A triple {a,b,c} is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # We iterate through all combinations of 3
    # To save time, we can be smarter, but N=128 is small enough for O(N^3) precomputation?
    # 128^3 / 6 ~ 350k checks. Each check is 3 LCM lookups. Very fast.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    # Bad triple {a, b, c}
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
                    
    # Helper: check if x can be added to set S
    def can_add(x, S):
        for a, b in conflicts[x]:
            if a in S and b in S:
                return False
        return True

    best_size = 0
    best_set = set()
    
    # Greedy strategies
    # 1. Natural order
    # 2. Reverse order
    # 3. Order by number of conflicts (ascending)
    # 4. Random order (multiple times)
    
    conflict_counts = {x: len(conflicts[x]) for x in elements}
    
    orderings = [
        elements,
        list(reversed(elements)),
        sorted(elements, key=lambda x: conflict_counts[x]),
        sorted(elements, key=lambda x: -conflict_counts[x])
    ]
    
    # Run greedy for each ordering
    for order in orderings:
        if time.time() > deadline - 5:
            break
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Local Search / Improvement
    # We will use the best_set as starting point for local search
    # and also try random restarts with local search.
    
    def local_search(S, time_limit):
        S = S.copy()
        current_size = len(S)
        
        while time.time() < time_limit:
            # 1. Try to add any element
            added = False
            # Iterate through elements not in S
            # To make it faster, we can just pick a random few or all
            # Since N=128, checking all is fine.
            for x in elements:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        current_size += 1
                        added = True
                        break # Try to add another one immediately
                        
            if added:
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
                continue # Keep trying to add
                
            # 2. Stuck. Try to swap (remove 1, add 1) to change composition
            # Or try Remove 1, Add 2.
            
            # Try Remove 1, Add 2
            # Pick a random y to remove
            # Then try to find x1, x2 to add
            
            # To avoid infinite loops, we can just do random swaps
            y = random.choice(list(S))
            S.remove(y)
            
            # Try to add one element
            added_one = False
            for x in elements:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        added_one = True
                        break
                        
            if added_one:
                # Now try to add another one
                for x in elements:
                    if x not in S:
                        if can_add(x, S):
                            S.add(x)
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()
                            break
            else:
                # Could not add any x, revert
                S.add(y)
                
            # If we didn't increase size, we might be stuck in a local max
            # We can also try simple swaps to escape
            # But the "Remove 1, Add 2" is a strong move.
            # If that fails, we might need to "shuffle" more.
            
            # Let's add a simple random swap if we haven't improved in a while?
            # For now, the loop above handles it.
            
            # To ensure we explore different y's, random.choice is good.
            
        return S, best_size, best_set

    # Run local search from best_set
    # And also from a few random greedy sets
    
    # Run local search on the current best
    S, best_size, best_set = local_search(best_set, deadline - 2)
    
    # If time remains, do random restarts
    while time.time() < deadline - 2:
        # Generate a random greedy set
        random_order = elements.copy()
        random.shuffle(random_order)
        S = set()
        for x in random_order:
            if can_add(x, S):
                S.add(x)
                
        # Run local search on it
        S, best_size, best_set = local_search(S, deadline - 2)
        
    return list(best_set)
