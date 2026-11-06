"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85
    n = 12
    
    def check_solution(arr):
        # Verify distinct subset sums
        sums = {0}
        for x in arr:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums |= new_sums
        return len(sums) == (1 << n)

    def exists_solution(M, current_time_limit):
        # Returns True if a valid set of n distinct positive integers <= M exists
        # Uses DFS
        arr = []
        sums = {0}
        
        def dfs(idx, current_max):
            if time.time() > current_time_limit:
                return False
            if idx == n:
                return True
                
            start_val = 1
            if idx > 0:
                start_val = arr[idx-1] + 1
                
            # We only care if a solution exists, so we can stop at first success
            for val in range(start_val, M + 1):
                # Check validity
                valid = True
                for s in sums:
                    if (s + val) in sums:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Recurse
                arr.append(val)
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + val)
                
                if dfs(idx + 1, val):
                    return True
                
                arr.pop()
                # Restore sums? No, we passed new_sums by value (set copy)
                # But we modified arr.
                # The sums variable in the parent scope is not modified.
                # Wait, `new_sums` is local. `sums` is from outer scope.
                # In the next iteration of the loop, `sums` is still the original set.
                # This is correct.
                
            return False
            
        return dfs(0, 0)

    # Binary search for minimum M
    # Lower bound: 12 (distinct positive integers)
    # Upper bound: 2048 (powers of 2)
    lo = 12
    hi = 2048
    
    # We need to be careful with the time limit.
    # Each check might take some time.
    # Let's allow ~5 seconds per check on average?
    # 90s total.
    # Binary search steps: log2(2048-12) ~ 11 steps.
    # So ~8s per check is safe.
    
    best_M = hi
    best_arr = None
    
    # First, ensure hi is valid
    # We know powers of 2 are valid, so hi=2048 is valid.
    
    # To save time, we can just do a linear search downwards from 2048?
    # No, binary search is better.
    
    # However, the DFS in exists_solution might be slow for small M if no solution exists.
    # For M < optimal, it will exhaust the search space.
    # The search space for M=1000 might be large.
    
    # Let's try to find the optimal M using a heuristic first?
    # Or just run the binary search with a strict time limit per check.
    
    # If a check times out, we assume it's False (conservative) or True (optimistic)?
    # If we assume False, we might miss the optimal M and return a larger one.
    # If we assume True, we might return an invalid M.
    # We must return a VALID array.
    # So if we are unsure, we should stick to a known valid M.
    
    # Strategy:
    # 1. Try to find a solution with M < 2048.
    # 2. Use DFS with a time limit.
    # 3. If found, record M and array.
    # 4. Try smaller M.
    
    # Let's iterate M downwards from 2048?
    # 2048, 2047, ...
    # If the optimal is close to 2048, this is fast.
    # If optimal is small, this is slow.
    
    # Given n=12, the optimal max is likely not tiny.
    # Powers of 2 sum to 4095.
    # If we use a denser set, the max can be smaller.
    # For n=12, minimal max is likely around 1000-1500?
    
    # Let's use the binary search but with a time limit per check.
    # If a check times out, we treat it as "False" (no solution found within time).
    # This might lead to returning a suboptimal (larger) M, but a valid one.
    # To ensure we return a valid one, we keep track of the best valid M found.
    
    current_best_M = hi
    current_best_arr = [1 << i for i in range(n)]
    
    # We know current_best_arr is valid.
    
    for _ in range(20): # Max binary search steps
        if time.time() > deadline - 10:
            break
            
        mid = (lo + hi) // 2
        
        # Check if solution exists with max <= mid
        # Give it a time limit
        check_start = time.time()
        check_limit = check_start + 5.0
        
        # We need a version of exists_solution that returns the array if found
        def find_solution(M, limit):
            arr = []
            sums = {0}
            
            def dfs(idx):
                if time.time() > limit:
                    return False
                if idx == n:
                    return True
                    
                start_val = 1
                if idx > 0:
                    start_val = arr[idx-1] + 1
                    
                for val in range(start_val, M + 1):
                    valid = True
                    for s in sums:
                        if (s + val) in sums:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    arr.append(val)
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + val)
                    
                    if dfs(idx + 1):
                        return True
                    
                    arr.pop()
                    
                return False
                
            if dfs(0):
                return arr[:]
            return None
            
        sol = find_solution(mid, check_limit)
        
        if sol:
            # Found a solution with max <= mid
            current_best_M = mid
            current_best_arr = sol
            hi = mid - 1
        else:
            # No solution found (or timeout)
            # If timeout, we don't know for sure.
            # To be safe, assume no solution if we don't find one quickly?
            # Or assume solution exists and try smaller?
            # If we assume solution exists, we might set hi to mid-1 and eventually
            # return a M that is actually invalid.
            # So we must only update current_best_arr if we find a valid solution.
            # If we don't find one, we just move lo up.
            lo = mid + 1
            
    return current_best_arr
# EVOLVE-BLOCK-END
