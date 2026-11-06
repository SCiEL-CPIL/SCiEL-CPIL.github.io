"""
Dissociated reciprocal set finder.
Strategy:
1. Use integer weights w_i = LCM(1..22) / i.
2. The condition that all subset sums of reciprocals are distinct is equivalent
   to all subset sums of weights being distinct.
3. Use a fast incremental check: maintain the set of all subset sums.
   When adding a new weight w, check if any (sum + w) collides with existing sums.
4. Use backtracking search to maximize cardinality, as greedy may not be optimal.
"""

import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

L = _lcm_list(range(1, 23))
WEIGHTS = {i: L // i for i in range(1, 23)}

def construct():
    """
    Find a dissociated reciprocal set from [1, 22] with maximum cardinality.
    """
    start_time = time.time()
    time_limit = 55  # Leave 5 seconds margin for safety
    
    # State for backtracking
    # We want to find the largest subset of [1, 22] that is dissociated.
    # We can use a recursive search with pruning.
    # To speed up, we precompute weights.
    
    # Greedy initial solution
    # Try to add elements in order of increasing weight? Or decreasing?
    # Smaller weights are "safer" as they cause smaller changes to sums?
    # Actually, larger weights are more likely to cause collisions if they are close to other sums.
    # Let's just do a standard greedy first to get a baseline.
    
    def greedy_check_order(order):
        S = []
        sums = {0}
        for x in order:
            w = WEIGHTS[x]
            # Check collision
            new_sums = []
            valid = True
            for s in sums:
                ns = s + w
                if ns in sums:
                    valid = False
                    break
                new_sums.append(ns)
            if valid:
                # Check if new sums collide with each other? 
                # No, if s1 != s2, s1+w != s2+w.
                # So we just need to check intersection with old sums.
                S.append(x)
                sums.update(new_sums)
        return S

    # Try a few orders to get a good initial bound
    best_S = []
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        sorted(range(1, 23), key=lambda x: WEIGHTS[x]),
        sorted(range(1, 23), key=lambda x: -WEIGHTS[x]),
    ]
    
    for order in orders:
        S = greedy_check_order(order)
        if len(S) > len(best_S):
            best_S = S
            
    best_size = len(best_S)
    
    # Backtracking search to find if we can do better
    # We search for a subset of size best_size + 1.
    # If found, increment best_size and repeat.
    
    # To make backtracking efficient, we can use the subset sum set.
    # But maintaining the full set in recursion is expensive to copy.
    # Instead, we can use a recursive function that takes the current set of sums as a frozenset?
    # Or we can just use the standard "check if valid" which is O(2^k).
    # Given k <= 20, 2^20 is 1M. Backtracking might be slow if we do full checks.
    
    # Optimized check:
    # We represent the set of sums as a Python set of integers.
    # We pass this set down. Copying a set of size 2^k takes O(2^k).
    # This is acceptable for k < 18.
    
    def is_valid_add(sums, w):
        # Check if adding w to the set of weights corresponding to 'sums' creates a collision.
        # sums contains all subset sums of the current set.
        # We need to check if {s + w for s in sums} intersects sums.
        for s in sums:
            if s + w in sums:
                return False
        return True

    def dfs(current_set, candidates, sums, depth, target_size):
        nonlocal best_S, best_size
        
        if time.time() - start_time > time_limit:
            return
        
        if len(current_set) == target_size:
            best_S = current_set[:]
            best_size = target_size
            return True
        
        if not candidates:
            return False
            
        # Pruning: if we can't reach target_size, return
        if len(current_set) + len(candidates) < target_size:
            return False
            
        for i in range(len(candidates)):
            x = candidates[i]
            w = WEIGHTS[x]
            
            if is_valid_add(sums, w):
                # Calculate new sums
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + w)
                
                # Recurse
                # Candidates remaining: candidates[i+1:]
                if dfs(current_set + [x], candidates[i+1:], new_sums, depth + 1, target_size):
                    return True
                    
        return False

    # Try to find a set of size best_size + 1
    # We need to choose an ordering of candidates for the search.
    # Randomized restarts might help escape local minima in the search space.
    import random
    
    max_iterations = 20
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle candidates to explore different branches
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        # We try to find a set of size best_size + 1
        target = best_size + 1
        # If target > 22, stop
        if target > 22:
            break
            
        # To speed up DFS, we can use a heuristic ordering in candidates?
        # Shuffling is fine for random restarts.
        
        found = dfs([], candidates, {0}, 0, target)
        if found:
            # We found a set of size target. Try to find target + 1
            continue
        else:
            # If we didn't find it in this shuffle, maybe try again?
            # Or accept that best_size is optimal?
            # With 20 restarts, we are fairly confident.
            pass

    return best_S

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
