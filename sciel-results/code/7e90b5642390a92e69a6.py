"""Difference triangle set (4,7) - local search with row rebuild."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_diffs(row)
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    def greedy_row(used_diffs, k, start_val=0):
        """Build a single row greedily given already-used differences."""
        row = [start_val]
        x = start_val
        while len(row) < k + 1:
            x += 1
            new_diffs = set(x - a for a in row)
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        return row
    
    def greedy_all(n_rows, k):
        """Build all rows greedily."""
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
    
    def rebuild_row(rows, used_diffs, row_idx, k):
        """Rebuild a specific row, keeping others fixed."""
        # Remove this row's diffs from used
        old_row = rows[row_idx]
        old_diffs = get_diffs(old_row)
        for d in old_diffs:
            used_diffs.discard(d)
        
        # Build new row greedily
        new_row = [0]
        x = 0
        while len(new_row) < k + 1:
            x += 1
            new = {x - a for a in new_row}
            if not new & used_diffs:
                new_row.append(x)
                used_diffs |= new
        
        rows[row_idx] = new_row
        return new_row
    
    def compute_max(rows):
        return max(max(row) for row in rows)
    
    # Initial greedy solution
    best_rows = greedy_all(n_rows, k)
    best_max = compute_max(best_rows)
    
    # Compute initial used diffs
    def compute_used(rows):
        used = set()
        for row in rows:
            used |= get_diffs(row)
        return used
    
    best_used = compute_used(best_rows)
    
    # Local search: try rebuilding individual rows
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Try rebuilding each row and keep if it improves
        improved = False
        for i in range(n_rows):
            if time.time() > deadline:
                break
            
            # Save state
            old_row = best_rows[i][:]
            old_used = best_used.copy()
            
            # Try rebuild
            new_row = rebuild_row(best_rows, best_used, i, k)
            new_max = compute_max(best_rows)
            
            if new_max < best_max:
                best_max = new_max
                improved = True
                # Keep the improvement
            else:
                # Revert
                best_rows[i] = old_row
                best_used = old_used
        
        if not improved:
            # No single-row rebuild helps; try random perturbations
            # Remove a random row and rebuild it with some randomness
            i = random.randint(0, n_rows - 1)
            old_row = best_rows[i][:]
            old_used = best_used.copy()
            
            # Try multiple random restarts for this row
            best_for_row = None
            best_for_row_max = best_max
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Save state
                temp_rows = [r[:] for r in best_rows]
                temp_used = best_used.copy()
                
                # Remove row i
                old_diffs = get_diffs(temp_rows[i])
                for d in old_diffs:
                    temp_used.discard(d)
                
                # Build with some randomness: occasionally skip a candidate
                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    x += 1
                    new = {x - a for a in new_row}
                    if not new & temp_used:
                        # With small probability, skip this and try next
                        if random.random() < 0.05 and x < 200:
                            continue
                        new_row.append(x)
                        temp_used |= new
                
                temp_rows[i] = new_row
                temp_max = compute_max(temp_rows)
                
                if temp_max < best_for_row_max:
                    best_for_row_max = temp_max
                    best_for_row = [r[:] for r in temp_rows]
                    best_for_row_used = temp_used.copy()
            
            if best_for_row is not None and best_for_row_max < best_max:
                best_rows = best_for_row
                best_used = best_for_row_used
                best_max = best_for_row_max
        
        # Occasionally try rebuilding multiple rows
        if iteration % 10 == 0 and time.time() < deadline:
            # Try rebuilding 2 random rows simultaneously
            if n_rows >= 2:
                i, j = random.sample(range(n_rows), 2)
                old_i = best_rows[i][:]
                old_j = best_rows[j][:]
                old_used = best_used.copy()
                
                # Remove both rows' diffs
                for d in get_diffs(best_rows[i]):
                    best_used.discard(d)
                for d in get_diffs(best_rows[j]):
                    best_used.discard(d)
                
                # Rebuild row i
                rebuild_row(best_rows, best_used, i, k)
                # Rebuild row j
                rebuild_row(best_rows, best_used, j, k)
                
                new_max = compute_max(best_rows)
                if new_max >= best_max:
                    # Revert
                    best_rows[i] = old_i
                    best_rows[j] = old_j
                    best_used = old_used
    
    return best_rows
