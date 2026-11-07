import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    valid_cache = {}
    
    def is_valid(c, sums_set):
        key = (c, frozenset(sums_set))
        if key in valid_cache:
            return valid_cache[key]
            
        new_sums = [c] + [c+s for s in sums_set]
        
        # Check against old sums
        for u in new_sums:
            # u divisible by old?
            for d in range(1, int(u**0.5) + 1):
                if u % d == 0:
                    if d in sums_set:
                        valid_cache[key] = False
                        return False
                    if d != u//d and u//d in sums_set:
                        valid_cache[key] = False
                        return False
            # old divisible by u?
            for m in range(2*u, 7700, u):
                if m in sums_set:
                    valid_cache[key] = False
                    return False
            
        # Check new vs new
        n = len(new_sums)
        for i in range(n):
            for j in range(i+1, n):
                a, b = new_sums[i], new_sums[j]
                if a < b:
                    if b % a == 0:
                        valid_cache[key] = False
                        return False
                elif b < a:
                    if a % b == 0:
                        valid_cache[key] = False
                        return False
                        
        valid_cache[key] = True
        return True

    def backtrack(candidates, current_set, current_sums):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        # Heuristic: if we have a large set, we can stop searching this branch?
        # No, we want max size.
        
        for i in range(len(candidates)):
            c = candidates[i]
            if is_valid(c, current_sums):
                new_sums = set(current_sums)
                new_sums.add(c)
                for s in current_sums:
                    new_sums.add(c+s)
                
                new_candidates = candidates[:i] + candidates[i+1:]
                backtrack(new_candidates, current_set + [c], new_sums)
                
                # Optional: if we found a very large set, we could prune more aggressively
                # e.g. if len(best_set) >= 15, we might not care about 14s.
                # But we don't know the max.
                
    # Initial run with descending order
    candidates = list(range(256, 0, -1))
    backtrack(candidates, [], set())
    
    # Random restarts
    # To avoid re-doing the same work, we can clear cache or keep it?
    # Keeping it is good.
    
    while time.time() < deadline - 5.0:
        random.shuffle(candidates)
        backtrack(candidates, [], set())
        
    return best_set
