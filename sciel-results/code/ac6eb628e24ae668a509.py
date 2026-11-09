import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_pairs(max_n):
    """
    Returns a list `conflicts` of size max_n+1.
    conflicts[x] is a list of tuples (a, b) with a < b < x 
    such that lcm(a,b) == lcm(a,x) == lcm(b,x).
    Note: We only need to check pairs (a,b) that are already in the set.
    Since we add elements in some order, we can just store all valid (a,b) pairs 
    for each x.
    """
    n = max_n
    # Precompute LCM table for speed
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
                
    conflicts = [[] for _ in range(n + 1)]
    
    # For each x, find all pairs (a, b) with a < b < x 
    # such that (a, b, x) is a bad triple.
    # Actually, the order in the set doesn't matter for validity, 
    # but for the greedy check "can we add x given current set S",
    # we need to know if any (a,b) in S forms a bad triple with x.
    # So for each x, we precompute ALL pairs (a,b) from [1, n] 
    # such that (a,b,x) is a bad triple.
    
    for x in range(1, n + 1):
        # a, b must be distinct from x and each other.
        # We can iterate a < b, and check if (a,b,x) is bad.
        # To avoid duplicates and ensure a,b < x is not required for precomputation,
        # we just find all pairs (a,b) in [1,n] x [1,n] with a!=b, b!=x, a!=x.
        # But checking all pairs for each x is O(N^3). 128^3 is fine.
        
        for a in range(1, n + 1):
            if a == x: continue
            for b in range(a + 1, n + 1):
                if b == x: continue
                # Check if (a, b, x) is a bad triple
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    conflicts[x].append((a, b))
                    
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    max_n = 128
    conflicts = _precompute_bad_pairs(max_n)
    
    # Convert conflicts to sets for faster lookup? 
    # conflicts[x] is a list of tuples. 
    # We will check if any (a,b) in conflicts[x] has both a,b in S.
    # S can be a set of ints.
    
    best_size = 0
    best_set = set()
    
    # Random restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate a random order
        order = list(range(1, max_n + 1))
        rng.shuffle(order)
        
        current_set = set()
        
        for x in order:
            # Check if x can be added
            valid = True
            for a, b in conflicts[x]:
                if a in current_set and b in current_set:
                    valid = False
                    break
            
            if valid:
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # Optional: Local search polish
            # Try to add any remaining elements? 
            # The greedy pass already tried all elements in 'order'.
            # But maybe a different order allows more?
            # The local search can try swaps.
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try to add any element not in set
                for v in range(1, max_n + 1):
                    if v in current_set:
                        continue
                    valid = True
                    for a, b in conflicts[v]:
                        if a in current_set and b in current_set:
                            valid = False
                            break
                    if valid:
                        current_set.add(v)
                        best_size = len(current_set)
                        best_set = current_set.copy()
                        improved = True
                        break # Restart loop after adding
                
                # If no single add works, try swaps (remove 1, add 1) 
                # to enable future adds? 
                # Or remove 2, add 3?
                # For now, single add is sufficient if we do multiple restarts.
                # But a swap might lead to a larger set.
                # Let's try a simple swap: remove u, add v.
                # This keeps size same, but might unblock an add.
                # We'll rely on restarts for now, as they are effective.
                
    return list(best_set)
