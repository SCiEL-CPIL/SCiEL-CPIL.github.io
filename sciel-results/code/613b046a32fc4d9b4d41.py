import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    # Lower bound for M
    # Based on problem description, simple counting bound is 112.
    min_m = 112
    
    def solve_for_M(M):
        # Backtracking to find rows of length k+1 with max val <= M
        # used_diffs contains ALL differences from all elements chosen so far
        # (both from completed rows and the current row being built)
        
        rows = []
        
        def backtrack(row_idx, current_row, used_diffs):
            if time.time() - start_time > time_limit - 2:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(list(current_row))
                # Move to next row, reset current_row to [0]
                # Note: used_diffs persists, as it contains diffs from this completed row
                success = backtrack(row_idx + 1, [0], used_diffs)
                if success:
                    return True
                else:
                    rows.pop()
                return False
            
            # Determine next value
            last_val = current_row[-1]
            # We need to place k - len(current_row) more elements after this one.
            # Min step is 1.
            max_next = M - (k - len(current_row))
            
            # Candidates are last_val + 1 to max_next
            # Randomize order to avoid bias
            candidates = list(range(last_val + 1, max_next + 1))
            random.shuffle(candidates)
            
            for v in candidates:
                # Calculate new differences involving v
                new_diffs = set()
                for x in current_row:
                    d = v - x
                    new_diffs.add(d)
                
                # Check for conflicts with existing used_diffs
                # used_diffs contains:
                # 1. Diffs from all completed rows.
                # 2. Diffs from elements already in current_row.
                # So if new_diffs intersects used_diffs, we have a conflict.
                if new_diffs & used_diffs:
                    continue
                
                # If no conflict, proceed
                # New used_diffs will include the new diffs
                new_used_diffs = used_diffs | new_diffs
                
                # Recurse
                if backtrack(row_idx, current_row + [v], new_used_diffs):
                    return True
            
            return False

        # Start backtracking
        # Initial state: row 0, current_row [0], used_diffs empty
        if backtrack(0, [0], set()):
            return rows
        else:
            return None

    # Iterate M from lower bound upwards
    for M in range(min_m, 200):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try a few times for each M to handle randomness in backtracking
        found = None
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts and time.time() - start_time < time_limit - 5:
            found = solve_for_M(M)
            if found:
                break
            attempts += 1
            
        if found:
            return found
            
    # Fallback: Greedy construction if backtracking fails
    # This ensures we return a valid solution
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            # Check against global used_diffs
            if not new_diffs & used_diffs:
                # Check against internal diffs of current row?
                # The greedy loop above doesn't track internal diffs separately.
                # However, if we add x, and x-a1 == x-a2, that's impossible for distinct a.
                # But x-a1 might equal a2-a3 (internal diff).
                # We need to check new_diffs against existing internal diffs too.
                
                # For simplicity in fallback, we just check against used_diffs (previous rows)
                # and assume internal consistency is handled by the fact that we pick increasing x?
                # No, internal conflicts are possible.
                # Let's do a proper check.
                
                # Compute current internal diffs
                curr_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        curr_diffs.add(row[j] - row[i])
                        
                if not (new_diffs & (used_diffs | curr_diffs)):
                    row.append(x)
                    # Update used_diffs with new_diffs AND curr_diffs?
                    # No, used_diffs should accumulate ALL diffs.
                    # So we add new_diffs and curr_diffs to used_diffs?
                    # Actually, curr_diffs are already "used" in the sense that they exist.
                    # We just need to ensure they don't conflict with future rows.
                    # So we should add them to used_diffs.
                    used_diffs |= new_diffs
                    used_diffs |= curr_diffs
                    x += 1
                else:
                    x += 1
            else:
                x += 1
        rows.append(row)
        
    return rows
