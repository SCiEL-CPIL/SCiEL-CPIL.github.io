import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    required = [1, 2, 5, 11]
    max_val = 284
    
    # Precompute all pairs sums for required
    req_sums = set()
    for i in range(len(required)):
        for j in range(i, len(required)):
            req_sums.add(required[i] + required[j])
            
    # Candidates are all numbers 1..284 not in required
    candidates = [x for x in range(1, max_val + 1) if x not in required]
    
    # We want to maximize size. Try 20, then 19, etc.
    # Max possible size is bounded by area. ~ sqrt(2*284) ~ 23.7. 
    # Given constraints, 20 is the theoretical max without constraints.
    # With constraints, 19 or 20 is likely.
    
    sys.setrecursionlimit(10000)
    
    def is_valid_addition(c, curr_sums, curr_set):
        # Check if c + y is in curr_sums for any y in curr_set
        # And 2*c is not in curr_sums
        if 2*c in curr_sums:
            return False
        for y in curr_set:
            if c + y in curr_sums:
                return False
        return True

    def dfs(curr_set, curr_sums, cand_idx, target_size):
        if len(curr_set) == target_size:
            return True
        if cand_idx == len(candidates):
            return False
            
        # Pruning: not enough candidates left to reach target
        remaining_slots = target_size - len(curr_set)
        remaining_cands = len(candidates) - cand_idx
        if remaining_slots > remaining_cands:
            return False
            
        # Time check
        if time.time() > deadline:
            return False
            
        # Try to add candidates
        # Heuristic: Try candidates in a specific order?
        # Randomization helps escape local minima, but DFS is deterministic.
        # We can shuffle candidates once per target size attempt?
        # Or use a fixed order that works well (e.g., increasing).
        
        # To improve performance, we can skip candidates that are "too close" to existing ones?
        # No, just iterate.
        
        for i in range(cand_idx, len(candidates)):
            c = candidates[i]
            if is_valid_addition(c, curr_sums, curr_set):
                # Add c
                curr_set.append(c)
                new_sums = set(curr_sums)
                # Add new sums
                new_sums.add(2*c)
                for y in curr_set[:-1]:
                    new_sums.add(c + y)
                
                if dfs(curr_set, new_sums, i + 1, target_size):
                    return True
                
                # Backtrack
                curr_set.pop()
                
        return False

    best_size = len(required)
    best_set = required[:]

    # Try to find larger sets
    # Start from 20 down to 5
    for target_size in range(20, len(required) + 1, -1):
        # Reset time limit? Or just use global deadline.
        # If we fail fast, we can try next.
        # We need to ensure we don't spend too much time on a failing target.
        # Let's allocate a fraction of time for each target.
        # 110s total.
        # If we try 20 and it takes 10s, we have 100s left.
        
        # Shuffling candidates might help find a solution faster if one exists,
        # but for DFS, the order matters for pruning.
        # Increasing order is standard.
        
        # Let's try a few random shuffles of candidates for each target size
        # to increase the chance of finding a solution within the time limit.
        attempts = 5
        found = False
        for _ in range(attempts):
            # Shuffle candidates
            import random
            random.shuffle(candidates)
            
            # Reset state
            curr_set = required[:]
            curr_sums = set(req_sums)
            
            if dfs(curr_set, curr_sums, 0, target_size):
                best_size = target_size
                best_set = curr_set[:]
                found = True
                break
        
        if found:
            break
            
    return sorted(best_set)
