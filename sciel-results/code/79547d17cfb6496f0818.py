import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_scope = float('inf')
    best_rows = None
    
    def try_construct(seed):
        random.seed(seed)
        used_diffs = set()
        rows = []
        
        # Local time limit for this attempt
        attempt_start = time.time()
        attempt_deadline = attempt_start + 1.0 # 1 second per attempt
        
        def backtrack(current_row, last_val):
            # Check time
            if time.time() > attempt_deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            x = last_val + 1
            # Prune: if x >= best_scope, we can't improve
            # Note: best_scope is from the outer scope, read-only here
            while x < best_scope:
                # Check validity
                is_valid = True
                for y in current_row:
                    if (x - y) in used_diffs:
                        is_valid = False
                        break
                
                if is_valid:
                    # Randomization: skip with small probability
                    # Use a higher probability to encourage diversity
                    if random.random() < 0.1:
                        x += 1
                        continue
                    
                    current_row.append(x)
                    new_diffs = {x - y for y in current_row[:-1]}
                    used_diffs.update(new_diffs)
                    
                    if backtrack(current_row, x):
                        return True
                    
                    # Backtrack
                    used_diffs.difference_update(new_diffs)
                    current_row.pop()
                
                x += 1
            return False

        for r in range(n_rows):
            row = [0]
            if not backtrack(row, 0):
                return None, float('inf')
            rows.append(row)
            
        scope = max(max(r) for r in rows)
        return rows, scope

    # Run multiple attempts
    attempt = 0
    while time.time() < deadline:
        rows, scope = try_construct(attempt)
        if rows is not None:
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                # If we hit the lower bound, we can stop
                if scope <= 105:
                    break
        attempt += 1
        
    if best_rows is None:
        # Fallback greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows

    return best_rows
