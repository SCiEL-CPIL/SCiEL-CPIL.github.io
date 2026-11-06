from fractions import Fraction
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55  # seconds, leave margin
    
    # Precompute LCM of 1..24
    # L = 5354228880
    L = 1
    for i in range(1, 25):
        L = L * i // math.gcd(L, i)
        
    # Weights w[i] = L // i
    w = [0] * 25
    for i in range(1, 25):
        w[i] = L // i
        
    # Function to check if a set is valid using integer weights
    def is_valid_set(S):
        if not S:
            return True
        # Compute subset sums of S
        sums = {0}
        for a in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + w[a])
            sums = sums | new_sums
        
        # Check condition for each a in S
        for a in S:
            # We need to check if w[a] is in subset sums of S \ {a}
            # Instead of recomputing subset sums for S \ {a}, 
            # we can check if there is a subset T of S \ {a} summing to w[a].
            # This is equivalent to: 
            # 1. w[a] is in subset sums of S (which it is, via {a})
            # 2. There is a representation NOT using a.
            
            # Simpler: Just compute subset sums of S \ {a}
            # This is O(2^k) per a, so O(k 2^k) total.
            # For k=15, this is ~200k ops. Fast.
            
            others = [b for b in S if b != a]
            target = w[a]
            
            # Subset sum check for 'others'
            # Use a set of reachable sums
            reachable = {0}
            for b in others:
                new_reach = set()
                for r in reachable:
                    nr = r + w[b]
                    if nr <= target: # Optimization: no need to go above target
                        new_reach.add(nr)
                reachable = reachable | new_reach
                if target in reachable:
                    return False
            
            # If loop finishes without returning False, target was not found
            # Wait, the logic above checks if target is reachable.
            # If it IS reachable, it's INVALID.
            # So if we find it, return False.
            # If we finish, it means it was NOT found, so this a is OK.
            
        return True

    # Backtracking to find max size
    best_size = 0
    best_set = []
    
    # We try to find a set of size k, starting from 24 down to 1
    # Or we can just do a DFS that maximizes size.
    
    # Let's do DFS that tries to include elements.
    # Order: 1 to 24.
    
    def dfs(idx, current_set, current_sums):
        nonlocal best_size, best_set
        
        if time.time() - start_time > timeout:
            return
            
        # Prune: if we can't beat best_size
        if len(current_set) + (24 - idx) <= best_size:
            return
            
        if idx > 24:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
            
        # Option 1: Skip idx
        dfs(idx + 1, current_set, current_sums)
        
        # Option 2: Include idx
        a = idx
        # Check if adding a is valid
        # 1. w[a] not in current_sums
        if w[a] in current_sums:
            return
            
        # 2. For all b in current_set, w[b] - w[a] not in subset_sums(current_set \ {b})
        valid = True
        for b in current_set:
            # Check if w[b] - w[a] is a subset sum of current_set \ {b}
            diff = w[b] - w[a]
            if diff < 0:
                continue
            if diff == 0:
                # This would imply w[b] == w[a], which is impossible for distinct a,b in 1..24
                continue
                
            # Check if diff is in subset sums of current_set \ {b}
            # We need to compute this.
            # Since current_set is small, we can just do a quick subset sum check.
            others = [c for c in current_set if c != b]
            # Subset sum check for 'others' to reach 'diff'
            reachable = {0}
            found = False
            for c in others:
                new_reach = set()
                for r in reachable:
                    nr = r + w[c]
                    if nr <= diff:
                        new_reach.add(nr)
                reachable = reachable | new_reach
                if diff in reachable:
                    found = True
                    break
            if found:
                valid = False
                break
                
        if not valid:
            return
            
        # Update sums for new set
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w[a])
            
        dfs(idx + 1, current_set + [a], new_sums)

    dfs(1, [], {0})
    
    if not best_set:
        # Fallback to single element
        return [1]
        
    return sorted(best_set)
