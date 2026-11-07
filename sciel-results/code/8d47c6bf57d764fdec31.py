import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    def check_addition(current_sums, x):
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        all_sums = current_sums.union(new_sums)
        if len(all_sums) != len(current_sums) + len(new_sums):
            return None
        
        all_sums_list = list(all_sums)
        for s in new_sums:
            for t in all_sums_list:
                if s == t:
                    continue
                if s > t:
                    if s % t == 0:
                        return None
                else:
                    if t % s == 0:
                        return None
        return all_sums

    def find_set_of_size(k, time_limit):
        if time.time() > time_limit:
            return None
            
        candidates = list(range(256, 0, -1))
        
        def dfs(current_elements, current_sums, remaining_candidates):
            if time.time() > time_limit:
                return None
                
            if len(current_elements) == k:
                return list(current_elements)
                
            if len(remaining_candidates) < k - len(current_elements):
                return None
                
            for i, x in enumerate(remaining_candidates):
                if time.time() > time_limit:
                    return None
                    
                new_sums = check_addition(current_sums, x)
                if new_sums is not None:
                    res = dfs(current_elements + [x], new_sums, remaining_candidates[i+1:])
                    if res is not None:
                        return res
                        
            return None

        # Attempt 1: Sorted descending
        res = dfs([], set(), candidates)
        if res is not None:
            return res
            
        # Attempt 2+: Randomized
        rng = random.Random()
        attempts = 0
        max_attempts = 50
        
        while time.time() < time_limit and attempts < max_attempts:
            cand = list(candidates)
            rng.shuffle(cand)
            res = dfs([], set(), cand)
            if res is not None:
                return res
            attempts += 1
            
        return None

    best_set = [256]
    current_size = 1
    
    # Try to find larger sets
    # We stop if we fail to find a set of size k
    for target_size in range(2, 11):
        if time.time() > deadline:
            break
            
        found = find_set_of_size(target_size, deadline)
        if found is not None and len(found) == target_size:
            best_set = found
            current_size = target_size
        else:
            break

    return [int(x) for x in best_set]
