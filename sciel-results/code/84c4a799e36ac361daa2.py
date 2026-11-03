import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Baseline
    rows = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in rows)
    best_rows = rows
    
    # 2. Improve
    current_scope = best_scope
    attempts = 0
    
    while current_scope > 0 and time.time() < deadline:
        # Try to find solution with this scope
        # Use random restarts if it fails?
        # Let's just run one search with randomized ordering.
        random.seed() # Reset seed for variety
        solution = backtracking_search(n_rows, k, current_scope, deadline)
        if solution:
            best_rows = solution
            best_scope = current_scope
            current_scope -= 1
        else:
            # If it failed, maybe it's impossible, or we ran out of time/nodes.
            # If we ran out of time, the loop condition will handle it.
            # If we just hit a dead end, we should try again with different randomness?
            # But the loop decrements scope. If we fail for M, we try M-1.
            # This is correct: if M is impossible, M-1 is also impossible?
            # No! If M is impossible, M+1 might be possible.
            # We are searching for the MINIMAL M.
            # So if we fail for M, we should stop?
            # No, we don't know if it's impossible or just hard to find.
            # But if we found a solution for M+1, and fail for M, it's likely M is impossible or we got unlucky.
            # Given the time limit, we should probably stop if we fail for a few consecutive M's?
            # Or just decrement and try.
            current_scope -= 1
            # To avoid infinite loop if greedy gave a huge scope and we can't find anything,
            # we should limit the number of failed attempts or time.
            # The deadline check handles time.
            
    return best_rows

def greedy_construct(n_rows, k):
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

def backtracking_search(n_rows, k, max_scope, deadline):
    used_mask = 0
    rows = []
    node_count = 0
    max_nodes = 100000 # Limit per search
    
    def backtrack(row_idx, current_row):
        nonlocal used_mask, node_count
        if time.time() > deadline or node_count > max_nodes:
            return False
        node_count += 1
        
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row[:])
            if backtrack(row_idx + 1, [0]):
                return True
            rows.pop()
            return False
        else:
            last = current_row[-1]
            candidates = []
            for x in range(last + 1, max_scope + 1):
                new_mask = 0
                conflict = False
                for a in current_row:
                    diff = x - a
                    bit = 1 << diff
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if conflict:
                    continue
                # Space check
                if max_scope - x < (k + 1 - (len(current_row) + 1)):
                    continue
                candidates.append((x, new_mask))
            
            random.shuffle(candidates)
            
            for x, new_mask in candidates:
                used_mask |= new_mask
                current_row.append(x)
                
                if backtrack(row_idx, current_row):
                    return True
                
                current_row.pop()
                used_mask ^= new_mask
                
            return False
            
    if backtrack(0, [0]):
        return rows
    return None
