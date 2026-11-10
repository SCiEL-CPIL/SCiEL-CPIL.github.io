import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds limit
    
    best_sequence = []
    
    # Greedy initialization to get a decent start
    values = []
    sums = set()
    for x in range(1, 257):
        tails = []
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        tails.append(x) # x itself
        # Check collision
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    best_sequence = values
    
    # DFS to try to find longer sequences
    # We can try to replace elements or just do a fresh search with heuristics
    
    def dfs(current_values, current_sums, last_val, depth):
        nonlocal best_sequence
        if time.time() > deadline:
            return
        
        if len(current_values) > len(best_sequence):
            best_sequence = current_values.copy()
        
        if last_val >= 256:
            return
            
        # Heuristic: Try values that are "good"
        # Start from last_val + 1 up to 256
        # To find long sequences, we might need to skip small values that cause collisions
        # or pick values that keep sums distinct
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            # Check if x can be added
            tails = []
            total = x
            for a in reversed(current_values):
                total += a
                tails.append(total)
            tails.append(x)
            
            if not current_sums.intersection(tails):
                # Valid
                new_values = current_values + [x]
                new_sums = current_sums.union(tails)
                dfs(new_values, new_sums, x, depth + 1)
                
    # Start DFS from empty
    # To speed up, we can start from the greedy result and try to extend or modify?
    # No, DFS from scratch with pruning is better for finding MAX length.
    # But DFS from scratch might get stuck in local minima if not guided well.
    # The greedy result is a lower bound.
    
    # Let's try to extend the greedy result?
    # The greedy result is a valid sequence.
    # We can try to insert new elements? No, sequence order is fixed by values.
    # We can try to replace elements?
    
    # Let's just run the DFS.
    # To improve speed, we can order candidates.
    # Maybe try larger values first?
    # If we try larger values first, we might find a short sequence quickly,
    # but we want the LONGEST.
    # So we should try to go deep.
    
    # Actually, the branching factor is high.
    # Let's use a smarter search:
    # We know the max length is likely small (< 20).
    # We can do a binary search on length L.
    # Can we find a sequence of length L?
    
    def can_find_length(L):
        # Backtracking to find sequence of length L
        pass
        
    # Given the time, let's stick to DFS with time limit.
    # To avoid exploring too many branches, we can prune if the maximum possible length
    # from current state is less than best_sequence length.
    # Max possible length:
    # Remaining values: 256 - last_val.
    # So max depth = len(current_values) + (256 - last_val).
    # If this is <= len(best_sequence), prune.
    
    def dfs_opt(current_values, current_sums, last_val):
        nonlocal best_sequence
        if time.time() > deadline:
            return
        
        current_len = len(current_values)
        if current_len > len(best_sequence):
            best_sequence = current_values.copy()
            
        # Pruning
        max_possible_len = current_len + (256 - last_val)
        if max_possible_len <= len(best_sequence):
            return
            
        if last_val >= 256:
            return

        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            # Pruning inside loop?
            # If x is too small, we might not reach the length.
            # But we just check max_possible_len at the top.
            
            tails = []
            total = x
            for a in reversed(current_values):
                total += a
                tails.append(total)
            tails.append(x)
            
            if not current_sums.intersection(tails):
                new_values = current_values + [x]
                new_sums = current_sums.union(tails)
                dfs_opt(new_values, new_sums, x)
                
    dfs_opt([], set(), 0)
    
    return best_sequence
