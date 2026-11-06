import time
import sys

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    n = 12
    # Initial best: powers of 2
    best_seq = [1 << i for i in range(n)]
    best_max = best_seq[-1]
    
    # State for DFS
    # sums: set of all subset sums
    # current_seq: list of chosen numbers
    # idx: next index to fill
    
    def is_valid(sums, x):
        # Check if x can be added to sums without collision
        # New sums: s + x for s in sums
        # Intersection with sums must be empty
        # Optimization: if x > max(sums), it's always valid (since sums are positive, min sum 0)
        # Actually, if x > max(sums), then min(new_sums) = x > max(sums) = max(old_sums).
        # So no overlap.
        if x > max(sums):
            return True
        
        # Check intersection
        # We can iterate through sums and check if s+x in sums
        # To speed up, we can check if (s+x) is in sums for any s in sums
        # Or, more efficiently, check if the set of (s+x) intersects sums.
        # Given small size (<= 4096), simple loop is fine.
        for s in sums:
            if s + x in sums:
                return False
        return True

    def dfs(idx, current_seq, sums, current_max):
        nonlocal best_seq, best_max
        
        # Time check
        if time.time() - start_time > time_limit:
            return False  # Signal to stop
        
        if idx == n:
            if current_max < best_max:
                best_max = current_max
                best_seq = current_seq[:]
            return True
        
        # Pruning: Even if we pick the smallest possible distinct integers for the rest,
        # can we beat best_max?
        # Next integers must be > current_max.
        # Smallest possible next is current_max + 1.
        # So min possible final max is current_max + (n - idx).
        if current_max + (n - idx) >= best_max:
            return False
        
        # Determine range for next element x
        # x must be > current_seq[-1] (distinct and sorted)
        # x must be valid with respect to sums
        # Upper bound for x: best_max (we want to improve)
        # Also, x cannot exceed best_max.
        
        start_x = current_seq[-1] + 1 if current_seq else 1
        end_x = best_max
        
        # Heuristic: Try smaller x first? Or larger?
        # Smaller x keeps sums "denser" but might block future options?
        # Larger x creates gaps.
        # For now, try from start_x to end_x.
        
        # Optimization: If start_x > max(sums), then any x >= start_x is valid.
        # In that case, we should pick the smallest x to keep max low.
        # If start_x > max(sums), pick x = start_x.
        # But wait, if we pick x = start_x, is it valid? Yes.
        # Then we can just add it and continue.
        # This greedy step is safe if x > max(sums).
        
        if start_x > max(sums):
            x = start_x
            new_sums = set(sums) | {s + x for s in sums}
            if dfs(idx + 1, current_seq + [x], new_sums, x):
                return True
            return False
        
        # Otherwise, we need to check validity for each x
        for x in range(start_x, end_x + 1):
            if not is_valid(sums, x):
                continue
            
            new_sums = set(sums) | {s + x for s in sums}
            
            # Recursive call
            # We pass a copy of current_seq? Or modify in place?
            # For simplicity and safety with recursion, pass copies or use backtracking.
            # Given n=12, copies are cheap.
            
            if dfs(idx + 1, current_seq + [x], new_sums, x):
                return True
        
        return False

    # Initial state
    initial_sums = {0, 1}
    initial_seq = [1]
    
    # Run DFS
    # We might need multiple restarts or different heuristics if the first path fails to improve quickly.
    # But let's try the basic DFS first.
    
    # To improve search, we can try random restarts or different ordering.
    # However, the structure of the problem suggests that the optimal solution is "close" to the greedy one.
    
    # Let's run the DFS.
    dfs(1, initial_seq, initial_sums, 1)
    
    return best_seq
