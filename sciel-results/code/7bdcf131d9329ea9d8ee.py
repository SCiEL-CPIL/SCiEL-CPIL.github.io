"""Difference triangle set (4,7) - Greedy construction with targeted local search."""
import random
import time

def _get_diffs(row):
    """Get all pairwise differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        row_diffs = _get_diffs(row)
        if all_diffs & row_diffs:
            return False
        all_diffs |= row_diffs
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)

def _greedy_initial(n_rows, k, seed):
    """Generate initial solution using greedy with randomization."""
    random.seed(seed)
    used_diffs = set()
    rows = []
    for r in range(n_rows):
        row = [0]
        # Random start to diversify
        x = 1 + random.randint(0, 20)
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used_diffs):
                row.append(x)
                used_diffs |= new_diffs
                # Randomly jump ahead to diversify
                x += random.randint(1, 10)
            else:
                x += 1
        rows.append(row)
    return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    timeout = 115
    
    best_rows = None
    best_score = float('inf')
    
    # Multiple restarts
    for restart in range(200):
        if time.time() - start_time > timeout:
            break
        
        # Generate initial solution
        current_rows = _greedy_initial(n_rows, k, restart)
        current_score = _max_entry(current_rows)
        
        # Local Search: Targeted reduction of max entry
        for _ in range(1000): # Max local search steps
            if time.time() - start_time > timeout:
                break
            
            # Find current max
            max_val = max(max(row) for row in current_rows)
            
            # Find which row/element has this max
            max_row_idx = -1
            max_elem_idx = -1
            for r_idx, row in enumerate(current_rows):
                for e_idx, val in enumerate(row):
                    if val == max_val:
                        max_row_idx = r_idx
                        max_elem_idx = e_idx
                        break
                if max_row_idx != -1:
                    break
            
            # Try to reduce the max element
            row = current_rows[max_row_idx]
            elem = max_elem_idx
            low = row[elem - 1] + 1
            high = max_val - 1 # We want to reduce it
            
            improved = False
            # Try random values in [low, high]
            for _ in range(30):
                if low > high:
                    break
                new_val = random.randint(low, high)
                old_val = row[elem]
                row[elem] = new_val
                
                if _is_valid(current_rows):
                    new_score = _max_entry(current_rows)
                    if new_score < max_val:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        # Valid but didn't reduce max? Revert
                        row[elem] = old_val
                else:
                    row[elem] = old_val
            
            if not improved:
                # Try to move other elements to free up space
                # Pick a random element in a random row
                r_idx = random.randint(0, n_rows - 1)
                e_idx = random.randint(1, k)
                
                old_val = current_rows[r_idx][e_idx]
                low = current_rows[r_idx][e_idx - 1] + 1
                high = current_rows[r_idx][e_idx + 1] - 1 if e_idx < k else old_val + 50
                
                if low >= high:
                    continue
                    
                new_val = random.randint(low, high)
                if new_val == old_val:
                    continue
                
                current_rows[r_idx][e_idx] = new_val
                
                if _is_valid(current_rows):
                    # Accept if it doesn't increase max
                    if _max_entry(current_rows) <= current_score:
                        pass # Keep
                    else:
                        current_rows[r_idx][e_idx] = old_val
                else:
                    current_rows[r_idx][e_idx] = old_val
            
            if current_score < best_score:
                best_score = current_score
                best_rows = [row[:] for row in current_rows]

    if best_rows is None:
        best_rows = _greedy_initial(n_rows, k, 0)
    
    return best_rows
