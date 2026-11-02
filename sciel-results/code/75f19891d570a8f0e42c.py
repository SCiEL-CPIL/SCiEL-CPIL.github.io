import sys
import time

def construct(n_rows, k):
    # k is such that row length is k+1
    # For "8 increasing integers", k=7.
    # Total differences per row = k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound for max value
    # We need total_diffs_needed distinct differences in [1, M]
    # So M >= total_diffs_needed
    lower_bound = total_diffs_needed
    
    # Upper bound from known results or just a safe limit
    # The prompt says best known is 134.
    # We'll search from lower_bound to lower_bound + 50
    # If not found, we might need to go higher, but 134 is the target.
    
    start_time = time.time()
    time_limit = 100.0  # Leave buffer for 120s limit
    
    def solve(max_val):
        # Backtracking solver
        # State:
        # rows: list of lists
        # used_diffs: set of int
        # row_idx: current row index
        # pos: current position in row (0 to k)
        # current_row: list of ints for current row
        
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx, pos, current_row):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                # Row complete
                rows.append(current_row[:])
                # Add diffs to used_diffs
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        used_diffs.add(current_row[j] - current_row[i])
                
                if backtrack(row_idx + 1, 1, [0]):
                    return True
                
                # Backtrack
                rows.pop()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        used_diffs.remove(current_row[j] - current_row[i])
                return False
            
            # Pruning: check if enough diffs remain
            remaining_rows = n_rows - row_idx
            if pos < k + 1:
                # Diffs needed for this row and rest
                # This is a rough estimate. 
                # Diffs needed for current row completion: 
                # If we place val at pos, we add (pos) new diffs (with 0..pos-1).
                # Total diffs for a full row is diffs_per_row.
                # Diffs already in current_row: pos*(pos-1)//2 ? No, pos is index.
                # If pos=1, we have [0, x1]. Diffs: x1-0. Count 1.
                # If pos=2, we have [0, x1, x2]. Diffs: x1, x2, x2-x1. Count 3.
                # General: number of diffs in a row of length L is L*(L-1)//2.
                # Here length is pos (since we start at 0 and add pos elements? No, current_row has pos elements if we count 0? 
                # Let's say current_row has length `pos`. Indices 0 to pos-1.
                # Diffs so far: pos*(pos-1)//2.
                # Diffs needed for this row to be complete (length k+1): diffs_per_row - pos*(pos-1)//2.
                # Diffs needed for remaining rows: (remaining_rows - 1) * diffs_per_row.
                # Total needed: diffs_per_row - pos*(pos-1)//2 + (remaining_rows - 1) * diffs_per_row.
                # Available: max_val - len(used_diffs).
                # If Available < Total Needed, prune.
                
                diffs_so_far_in_row = pos * (pos - 1) // 2
                needed_current = diffs_per_row - diffs_so_far_in_row
                needed_rest = (remaining_rows - 1) * diffs_per_row
                total_needed = needed_current + needed_rest
                available = max_val - len(used_diffs)
                
                if available < total_needed:
                    return False
            
            last_val = current_row[-1]
            
            # Try candidates from last_val + 1 to max_val
            # Heuristic: try smaller values first? Or larger?
            # Smaller values keep the max low, but might block future larger values?
            # Actually, we are constrained by max_val.
            # Let's try smaller values first to satisfy the "minimize max" spirit, 
            # though we are just checking existence for a fixed max_val.
            
            for val in range(last_val + 1, max_val + 1):
                # Check conflicts
                valid = True
                new_diffs = []
                for u in current_row:
                    d = val - u
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Apply
                current_row.append(val)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos + 1, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False
        
        # Start backtracking
        # First element is always 0
        if backtrack(0, 1, [0]):
            return rows
        return None

    best_solution = None
    for M in range(lower_bound, lower_bound + 100):
        solution = solve(M)
        if solution:
            best_solution = solution
            break
        if time.time() - start_time > time_limit:
            break
            
    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if backtracking fails or times out
        # This ensures we return a valid solution even if not optimal
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
        return rows
