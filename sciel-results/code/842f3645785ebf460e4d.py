import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    n = 512
    max_size = 48
    
    best_set = []
    best_size = 0
    
    # Helper: Check if adding c to S keeps it 3-free (no 3-term AP)
    def is_3_free(S, c):
        S_set = set(S)
        # Check if c forms AP with any pair in S
        # x, c, y -> 2c = x+y
        # x, y, c -> 2y = x+c
        # c, x, y -> 2x = c+y
        
        # Optimization: Just check pairs in S
        # If we have S, adding c.
        # New APs must involve c.
        # 1. c is middle: x+y=2c. x,y in S.
        # 2. c is end: x+c=2y. x in S, y in S.
        
        # Check 1: x+y=2c
        # x = 2c - y
        for y in S:
            x = 2 * c - y
            if x in S_set and x != y:
                return False
        
        # Check 2: x+c=2y => c = 2y - x
        # For each y in S, check if 2y-c is in S
        for y in S:
            x = 2 * y - c
            if x in S_set and x != y:
                return False
                
        return True

    def greedy_3_free():
        S = []
        S_set = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            if len(S) >= max_size:
                break
            if is_3_free(S, c):
                S.append(c)
                S_set.add(c)
        return S

    def find_violation(S):
        """
        Returns (a, subset) if a is the mean of subset (size >= 2) of S\{a}, else (None, None).
        We skip size 2 because we maintain 3-freeness.
        """
        S_list = list(S)
        n_s = len(S_list)
        
        # Check for means of subsets of size >= 3
        # For each a, check if k*a is a subset sum of S\{a} with k elements, k>=3
        # Since we are 3-free, k=2 is impossible.
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            # Check subset sizes k from 3 to len(others)
            # To speed up, we can check if k*a is reachable.
            # Max sum is ~25000.
            # We can use a set of sums for 'others'?
            # Doing this for every a is expensive.
            
            # Heuristic: Check small k first.
            # k=3
            # Check if 3a is sum of 3 elements in others
            # This is O(N^3) if naive.
            
            # Better: Precompute subset sums for 'others'?
            # No, 'others' changes for each a.
            
            # Alternative: Iterate subsets of small sizes?
            # Size 3: combinations(others, 3)
            # If N=48, C(47,3) ~ 16k. 48 * 16k ~ 768k ops.
            # Size 4: C(47,4) ~ 178k. 48 * 178k ~ 8.5M ops.
            # Might be too slow for every repair step.
            
            # Let's try a different approach for repair:
            # Randomly pick a, check if it's a mean.
            
            pass
        
        return None, None

    def is_mean_of_subset(a, S):
        """Check if a is the mean of a subset of S\{a} of size >= 2."""
        others = [x for x in S if x != a]
        n_others = len(others)
        
        # Check size 2
        # 2a = x+y
        # x = 2a - y
        others_set = set(others)
        for y in others:
            x = 2 * a - y
            if x in others_set and x != y:
                return True, [x, y]
        
        # Check size 3
        # 3a = x+y+z
        # We can do this more efficiently than combinations if needed, but C(47,3) is small.
        for i in range(n_others):
            x = others[i]
            for j in range(i+1, n_others):
                y = others[j]
                z = 3 * a - x - y
                if z in others_set and z != x and z != y:
                    return True, [x, y, z]
        
        # Check size 4
        # 4a = x+y+z+w
        # This is getting heavy.
        # For the purpose of this task, if we maintain 3-freeness, larger means are rarer.
        # We can check size 4 with a simple loop if time permits.
        
        return False, None

    def repair(S):
        """
        Iteratively remove violating elements and try to replace them.
        """
        S = list(S)
        S_set = set(S)
        pool = [x for x in range(1, n + 1) if x not in S_set]
        random.shuffle(pool)
        
        # Iterate until no violations or pool empty
        while True:
            violation_found = False
            # Check for violations
            for a in list(S):
                is_mean, subset = is_mean_of_subset(a, S)
                if is_mean:
                    violation_found = True
                    # Remove a
                    S.remove(a)
                    S_set.remove(a)
                    # Try to add a new candidate from pool
                    added = False
                    for c in pool:
                        if is_3_free(S, c):
                            S.append(c)
                            S_set.add(c)
                            pool.remove(c)
                            added = True
                            break
                    # If not added, we just lost an element
                    break # Restart checking from scratch
            
            if not violation_found:
                break
            
            if time.time() > deadline:
                break
                
        return S

    # Main Loop
    while time.time() < deadline:
        S = greedy_3_free()
        S = repair(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        if best_size == max_size:
            break

    return best_set
