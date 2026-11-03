import time
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Greedy Initialization
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, max(max(r) for r in rows)

    rows_greedy, scope_greedy = greedy()
    best_rows = rows_greedy
    best_scope = scope_greedy
    
    if n_rows == 1:
        return rows_greedy

    # 2. Backtracking Search
    current_rows = []
    used_diffs = set()
    
    # Lex status: 0 = undecided, 1 = greater, -1 = less
    # We pass this down or compute it.
    # Since we only care about the current row vs prev_row, we can compute it on the fly.
    
    def dfs(row_idx, current_row, lex_status):
        nonlocal best_scope, best_rows, deadline
        
        if time.time() > deadline:
            return
            
        # Pruning based on current max
        if current_row and current_row[-1] >= best_scope:
            return

        if row_idx == n_rows:
            current_scope = max(max(r) for r in current_rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
            return

        if not current_row:
            current_row = [0]
            # Lex status for new row vs prev row
            # prev_row[0] is 0. current_row[0] is 0.
            # So status is 0 (undecided) if row_idx > 0, else 0 (irrelevant)
            if row_idx > 0:
                lex_status = 0
            else:
                lex_status = 1 # No constraint for first row, treat as "satisfied" to skip checks
        
        if len(current_row) == k + 1:
            # Row complete
            if row_idx > 0:
                if lex_status != 1:
                    # Must be strictly greater. 
                    # If status is 0, it means curr == prev, which is invalid.
                    # If status is -1, it's invalid.
                    return
            
            current_rows.append(current_row[:])
            dfs(row_idx + 1, [], 0) # Reset status for next row
            current_rows.pop()
            return

        last_mark = current_row[-1]
        min_next = last_mark + 1
        max_next = best_scope - 1
        
        if min_next > max_next:
            return

        prev_row = current_rows[-1] if row_idx > 0 else None
        
        if prev_row is not None and lex_status == 0:
            # We are still undecided. 
            # The next mark MUST be > prev_row[len(current_row)]
            # Note: len(current_row) < k+1
            min_next = max(min_next, prev_row[len(current_row)] + 1)
            
            # If min_next > max_next, prune
            if min_next > max_next:
                return

        # Iterate
        for x in range(min_next, max_next + 1):
            if time.time() > deadline:
                return
            
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Determine new lex status
            new_status = lex_status
            if prev_row is not None:
                if lex_status == 0:
                    # Compare x with prev_row[len(current_row)]
                    if x > prev_row[len(current_row)]:
                        new_status = 1
                    elif x < prev_row[len(current_row)]:
                        new_status = -1
                    else:
                        new_status = 0
                # If lex_status was 1, it stays 1.
                # If lex_status was -1, we would have pruned earlier.
            
            # If new_status is -1, skip
            if new_status == -1:
                continue

            current_row.append(x)
            used_diffs |= new_diffs
            
            dfs(row_idx, current_row, new_status)
            
            current_row.pop()
            used_diffs -= new_diffs
            
            if x >= best_scope:
                break

    dfs(0, [], 1)
    
    return best_rows
