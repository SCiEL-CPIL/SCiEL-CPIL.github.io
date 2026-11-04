"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _is_valid(rows, n_rows, k):
    """Check that all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_row(used, k, start_x=1, max_x=500):
    """
    Greedily build a single row starting at 0 with k+1 entries.
    Tries to find the smallest possible next mark.
    Returns the row list or None if it fails.
    """
    row = [0]
    x = start_x
    while len(row) < k + 1:
        if x > max_x:
            return None
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used.update(new_diffs)
            x += 1
        else:
            x += 1
    return row

def _greedy_rows(n_rows, k, max_x=500):
    """
    Build n_rows rows greedily.
    Returns list of rows or None.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, max_x=max_x)
        if row is None:
            return None
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, time_limit=10.0):
    """
    Try to improve the scope by local search.
    We'll try to reduce the maximum entry by re-optimizing rows.
    """
    start = time.time()
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Try to improve by rebuilding rows in different orders
    # and with different greedy strategies
    
    # Strategy 1: Try different orderings of which differences to use first
    # Strategy 2: For each row, try to find a shorter version
    
    # Simple improvement: try to reduce the last element of the last row
    # by trying alternative endings
    
    # More effective: iterative improvement
    # Try swapping or adjusting individual marks
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try to reduce scope by modifying the row with the largest entry
        max_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
        max_row = best_rows[max_idx]
        current_max = max(max_row)
        
        # Try to find a replacement for the last few elements
        # that has a smaller maximum
        
        # Compute used differences from all other rows
        used_by_others = set()
        for i, row in enumerate(best_rows):
            if i == max_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    used_by_others.add(row[b] - row[a])
        
        # Try to rebuild this row with a smaller maximum
        target_max = current_max - 1
        while target_max >= 1:
            # Try greedy with a cap
            row = [0]
            x = 1
            success = True
            while len(row) < k + 1:
                if x > target_max:
                    success = False
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used_by_others:
                    row.append(x)
                    x += 1
                else:
                    x += 1
            if success:
                # Check that this row's diffs don't conflict with others
                row_diffs = set()
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        d = row[b] - row[a]
                        if d in used_by_others:
                            success = False
                            break
                        row_diffs.add(d)
                    if not success:
                        break
                if success:
                    best_rows[max_idx] = row
                    new_scope = _scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
            target_max -= 1
        
        if not improved:
            # Try a different approach: random restart with perturbation
            # Pick a random row and try to rebuild it with random choices
            idx = random.randint(0, n_rows - 1)
            used_by_others = set()
            for i, row in enumerate(best_rows):
                if i == idx:
                    continue
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        used_by_others.add(row[b] - row[a])
            
            # Try several random restarts for this row
            for attempt in range(20):
                row = [0]
                x = 1
                max_val = best_scope
                success = True
                while len(row) < k + 1:
                    if x > max_val:
                        success = False
                        break
                    # With some probability, skip a value to explore alternatives
                    if random.random() < 0.1 and x < max_val:
                        x += 1
                        continue
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_by_others:
                        row.append(x)
                        x += 1
                    else:
                        x += 1
                if success:
                    row_diffs = set()
                    valid = True
                    for a in range(len(row)):
                        for b in range(a + 1, len(row)):
                            d = row[b] - row[a]
                            if d in used_by_others or d in row_diffs:
                                valid = False
                                break
                            row_diffs.add(d)
                        if not valid:
                            break
                    if valid and max(row) < max(best_rows[idx]):
                        best_rows[idx] = row
                        new_scope = _scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        break
    
    return best_rows, best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with a greedy construction
    rows = _greedy_rows(n_rows, k, max_x=500)
    if rows is None:
        # Fallback: very simple construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                x += 1
            rows.append(row)
    
    current_scope = _scope(rows)
    
    # Try local search to improve
    if time.time() < deadline - 15:
        rows, current_scope = _local_search(rows, n_rows, k, time_limit=min(30.0, deadline - time.time() - 10))
    
    # Try multiple random restarts if time permits
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    while time.time() < deadline - 5:
        # Random restart: build rows with randomized greedy
        used = set()
        new_rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 1
            max_x = best_scope + 10
            while len(row) < k + 1:
                if x > max_x:
                    success = False
                    break
                # Randomized: sometimes skip
                if random.random() < 0.05:
                    x += 1
                    continue
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    x += 1
                else:
                    x += 1
            if not success:
                break
            new_rows.append(row)
        
        if success and len(new_rows) == n_rows:
            scope = _scope(new_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in new_rows]
                # Do local search on this improved solution
                if time.time() < deadline - 10:
                    best_rows, best_scope = _local_search(best_rows, n_rows, k, time_limit=min(10.0, deadline - time.time() - 5))
        
        # If we've been stuck, try a more aggressive approach
        # Rebuild from scratch with different parameters
        if random.random() < 0.3:
            used = set()
            new_rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                # Start from a random offset in the greedy process
                x = random.randint(1, 5)
                while len(row) < k + 1:
                    if x > best_scope + 20:
                        success = False
                        break
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used.update(new_diffs)
                    x += 1
                if not success:
                    break
                new_rows.append(row)
            
            if success and len(new_rows) == n_rows:
                scope = _scope(new_rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in new_rows]
    
    # Final validation
    if not _is_valid(best_rows, n_rows, k):
        # Fallback to a guaranteed valid construction
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                x += 1
            best_rows.append(row)
    
    return best_rows
# EVOLVE-BLOCK-END
