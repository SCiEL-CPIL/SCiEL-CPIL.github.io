import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Get initial greedy solution (Seed)
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    initial_rows = greedy_construct()
    initial_scope = max(max(r) for r in initial_rows)
    best_rows = initial_rows
    best_scope = initial_scope
    
    # 2. Try to improve
    # Total diffs needed
    total_diffs = n_rows * (k + 1) * k // 2
    min_m = total_diffs
    
    # We search M from min_m to best_scope - 1
    # To avoid hanging on impossible low M, we limit time per M.
    
    # Use bitmask for used diffs if M is small enough
    # M < 200 is safe for Python int bit ops or bytearray
    
    def backtrack(i, used_mask, current_rows, M):
        nonlocal best_scope, best_rows
        
        if time.time() > deadline:
            return False
            
        if i == n_rows:
            scope = max(max(r) for r in current_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in current_rows]
            return True
        
        # Generate candidate rows
        # We want rows with max <= M
        # Randomized search for a valid row
        
        # To speed up, we can try a few random permutations of the "greedy" choices?
        # Or just random rows.
        
        attempts = 0
        # Limit attempts to avoid infinite loops on hard instances
        max_attempts = 2000 
        while attempts < max_attempts:
            attempts += 1
            
            # Generate a random row with max <= M
            # Heuristic: Start with 0, pick next randomly from 1..M
            # But ensure it's a valid Golomb ruler
            row = [0]
            used_in_row = 0 # bitmask for diffs within this row
            valid = True
            current_max = 0
            
            for _ in range(k):
                # Try to pick a new point
                found = False
                # Randomly pick from 1..M
                # To bias towards small, we can pick from 1..M with replacement?
                # No, just uniform.
                
                # Optimization: Try values in a random order?
                # Or just try a few random values.
                
                for _ in range(50): # 50 tries per point
                    x = random.randint(1, M)
                    if x <= current_max:
                        continue
                    
                    # Check conflicts with existing points in row
                    conflict = False
                    new_diffs_mask = 0
                    for p in row:
                        d = x - p
                        # d is in 1..M
                        bit = 1 << d
                        if used_in_row & bit:
                            conflict = True
                            break
                        new_diffs_mask |= bit
                    
                    if not conflict:
                        row.append(x)
                        current_max = x
                        used_in_row |= new_diffs_mask
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check global conflict
            # used_mask is global
            if used_mask & used_in_row:
                continue
            
            # Recurse
            new_mask = used_mask | used_in_row
            if backtrack(i + 1, new_mask, current_rows + [row], M):
                return True
        
        return False

    # Search M
    # We start from min_m.
    # If min_m is much lower than best_scope, we might waste time.
    # But we want the best score.
    
    # Let's try a few values around the known best (112) if k=5?
    # No, keep it generic.
    
    # To be efficient, we can do a binary search? No, monotonicity of "solvable" is not guaranteed to be simple (if M works, M+1 works).
    # So linear search from min_m is correct.
    
    # However, if min_m is 105 and best is 112, we only check 105..111.
    
    m = min_m
    while m < best_scope and time.time() < deadline:
        # Try to find a solution for M=m
        # Run a few restarts
        found = False
        restarts = 0
        # Time limit per M: 5 seconds
        m_start = time.time()
        while not found and restarts < 3 and (time.time() - m_start) < 5.0 and time.time() < deadline:
            restarts += 1
            if backtrack(0, 0, [], m):
                found = True
                break
        
        if found:
            # Found a solution for m
            # Since we are iterating upwards, this is the minimal m found so far.
            # We can break if we assume m is optimal?
            # Yes, because we started from min_m.
            break
        
        m += 1
        
    return best_rows
# EVOLVE-BLOCK-END
