import time
import random

def _is_valid_addition(current_sums, x):
    """
    Check if x can be added to a set with subset sums 'current_sums'.
    current_sums includes 0.
    Returns (True, new_sums) if valid, (False, None) otherwise.
    """
    new_sums = set()
    for s in current_sums:
        ns = s + x
        # Collision check: ns must not be in current_sums
        if ns in current_sums:
            return False, None
        # Collision check: ns must be unique within new_sums
        if ns in new_sums:
            return False, None
        new_sums.add(ns)
    
    # Divisibility check
    # A = current_sums - {0}
    # B = new_sums - {0}
    A = [s for s in current_sums if s > 0]
    B = [s for s in new_sums if s > 0]
    
    # Check A vs B
    for a in A:
        for b in B:
            if a == b:
                return False, None # Should be caught by collision, but just in case
            if a < b:
                if b % a == 0:
                    return False, None
            else:
                if a % b == 0:
                    return False, None
                    
    # Check B vs B
    # We can sort B to make it efficient
    B_sorted = sorted(B)
    for i in range(len(B_sorted)):
        bi = B_sorted[i]
        # bi cannot divide any bj where j > i
        for j in range(i + 1, len(B_sorted)):
            bj = B_sorted[j]
            if bj % bi == 0:
                return False, None
                
    return True, new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = [31, 32, 34] # Fallback
    
    # Try to find a set of size K
    # Max possible size is 8 due to 256 limit and superincreasing requirement.
    # We try K=8, then 7, etc.
    
    candidates = list(range(1, 257))
    
    for K in range(8, 2, -1):
        found = False
        # To make search faster, we can try different orderings or random restarts
        # Let's try a DFS with pruning.
        # We will iterate candidates in descending order to favor large sums.
        # Or we can use a heuristic: start with large numbers.
        
        def dfs(idx, current_set, current_sums):
            nonlocal found, best_set
            if time.time() > deadline:
                return
            if len(current_set) == K:
                found = True
                best_set = current_set[:]
                return
            if found:
                return
                
            # Pruning: if we don't have enough candidates left, return
            remaining = len(candidates) - idx
            if remaining < K - len(current_set):
                return
                
            for i in range(idx, len(candidates)):
                if time.time() > deadline:
                    return
                c = candidates[i]
                valid, new_sums = _is_valid_addition(current_sums, c)
                if valid:
                    current_set.append(c)
                    dfs(i + 1, current_set, new_sums)
                    current_set.pop()
                    if found:
                        return
                    if time.time() > deadline:
                        return

        # Run DFS
        # We might need to restart with different candidate orderings if the first one fails
        # Let's try descending order first
        candidates_desc = sorted(candidates, reverse=True)
        
        # Try a few random shuffles if descending fails quickly?
        # No, DFS is deterministic. If it fails, it fails for that order.
        # We can try a limited number of random restarts for each K.
        
        attempts = 0
        max_attempts = 20
        while attempts < max_attempts and not found and time.time() < deadline:
            if attempts == 0:
                # First attempt: descending
                current_candidates = candidates_desc
            else:
                # Subsequent attempts: random shuffle, but bias towards larger numbers?
                # Or just random.
                current_candidates = candidates[:]
                random.shuffle(current_candidates)
            
            dfs(0, [], {0})
            attempts += 1
            
            if found:
                break
            
        if found:
            break
            
    # Ensure return type is list of ints
    return [int(x) for x in best_set]

if __name__ == "__main__":
    construct()
