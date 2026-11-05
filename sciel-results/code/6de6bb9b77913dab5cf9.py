import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    # Helper to check validity and get diffs
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # Greedy baseline
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
        return rows

    best_rows = greedy()
    best_scope = max(max(row) for row in best_rows)
    
    # Backtracking search
    # We will try to find a solution with scope < best_scope
    # We use a recursive function with backtracking
    # To avoid deep recursion limits and to allow time checking, we use a iterative deepening or just a DFS with a node limit/time limit.
    
    # Since k and n_rows are small (typically k=8, n_rows=4 in the example, but could be larger),
    # the search space is manageable with good pruning.
    
    # Let's use a DFS that tries to build rows one by one.
    # State: current_row, rows_completed, used_diffs
    # We want to minimize the max value.
    
    # We'll use a global variable to track the best solution found
    # and a timer to stop.
    
    def backtrack(row_idx, current_row, used_diffs, max_val_so_far):
        nonlocal best_rows, best_scope, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # All rows completed
            scope = max(max(r) for r in best_rows_temp) # Wait, we need to pass the rows
            # Actually, we can just update best if the current max_val_so_far is smaller
            # But max_val_so_far is the max of the last row? No, it's the max of all rows so far.
            # Let's pass the list of rows.
            pass
        
    # This recursive approach might be slow if not pruned well.
    # Let's try a different strategy: **Local Search / Simulated Annealing**
    # It is often more effective for this type of combinatorial optimization.
    
    def local_search():
        nonlocal best_rows, best_scope, start_time, time_limit
        
        # Start from greedy solution
        rows = [row[:] for row in best_rows]
        used_diffs = set()
        for row in rows:
            used_diffs |= get_diffs(row)
        
        current_scope = best_scope
        temperature = 100.0
        cooling = 0.99
        min_temperature = 0.1
        
        while temperature > min_temperature and time.time() - start_time < time_limit:
            improved = False
            # Try to improve
            for _ in range(1000): # Fixed number of moves per temperature step
                if time.time() - start_time > time_limit:
                    return
                
                # Pick a random row and a random position (not 0)
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k) # 1 to k (indices in the row)
                
                old_val = rows[r_idx][pos]
                
                # Try to find a smaller value that is valid
                # The new value must be > rows[r_idx][pos-1] and < old_val
                # Also, the new differences must not conflict with used_diffs (excluding the old ones)
                
                # Remove old differences involving this element
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - rows[r_idx][i])
                for j in range(pos + 1, len(rows[r_idx])):
                    old_diffs.add(rows[r_idx][j] - old_val)
                
                # Temporarily remove them from used_diffs
                for d in old_diffs:
                    used_diffs.discard(d)
                
                # Try smaller values
                found = False
                for new_val in range(rows[r_idx][pos-1] + 1, old_val):
                    # Check if new_val is valid
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - rows[r_idx][i])
                    for j in range(pos + 1, len(rows[r_idx])):
                        new_diffs.add(rows[r_idx][j] - new_val)
                    
                    if not new_diffs & used_diffs:
                        # Valid move
                        # Accept if it reduces scope or with probability
                        new_scope = max(max(row) for row in rows)
                        if new_scope < current_scope:
                            # Always accept
                            rows[r_idx][pos] = new_val
                            used_diffs |= new_diffs
                            current_scope = new_scope
                            improved = True
                            found = True
                            break
                        else:
                            # Accept with probability
                            if random.random() < 0.1: # High acceptance rate for exploration
                                rows[r_idx][pos] = new_val
                                used_diffs |= new_diffs
                                found = True
                                break
                
                if not found:
                    # Restore old differences
                    used_diffs |= old_diffs
                else:
                    # If we rejected, restore the old state
                    rows = old_rows
            # Update best if improved
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in rows]
            
            temperature *= cooling
    
    local_search()
    
    return best_rows

construct(4, 8)
