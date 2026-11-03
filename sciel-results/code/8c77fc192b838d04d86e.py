import time
import random
import sys

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for d in _compute_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, skip_prob=0.0):
    """
    Greedy initialization.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # With some probability, skip to create diversity
                if skip_prob > 0 and random.random() < skip_prob:
                    # Skip a few valid values
                    skipped = False
                    for _ in range(random.randint(1, 5)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        # If we skipped too far and hit a conflict or limit, just take the first valid
                        # But we already incremented x, so we might need to backtrack or just take x if valid
                        # Actually, the loop above breaks on first valid. If none found, we just continue outer loop
                        # But we must append something. Let's just take the first valid we found before skipping?
                        # Simpler: just append x if valid, else continue.
                        # The logic above is a bit messy. Let's simplify:
                        pass 
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _rebuild_row_dfs(forbidden, k, max_scope_hint, deadline):
    """
    Rebuild a row to minimize its maximum element using DFS with backtracking.
    forbidden: set of differences already used by other rows.
    Returns the best row found or None.
    """
    best_row = None
    best_last = max_scope_hint + 10
    
    # We want to find a row [0, r1, ..., rk] such that all diffs are not in forbidden.
    # And we want to minimize rk.
    
    # To speed up, we can use a simple DFS that tries small values first.
    # We limit the search to avoid explosion.
    
    def dfs(row, local_used):
        nonlocal best_row, best_last
        if time.time() > deadline:
            return None
        
        if len(row) == k + 1:
            if row[-1] < best_last:
                best_last = row[-1]
                best_row = list(row)
            return
        
        last = row[-1]
        # Try next values
        # We can estimate a lower bound for the next value.
        # The smallest possible next value is last + 1.
        # The largest possible next value is best_last - (k - len(row))
        # because we need at least 1 step for each remaining element.
        
        min_next = last + 1
        # If we are trying to beat best_last, the current element must be <= best_last - (k - len(row))
        # But we are building forward.
        
        # Heuristic: try values close to last first? Or spread out?
        # For difference sets, spreading out is often necessary to avoid small differences.
        # However, greedy smallest-first usually finds a valid row quickly if one exists.
        
        # Let's try a range of values.
        # To avoid infinite loops or too much time, we limit the number of tries per level.
        
        # A better heuristic: The differences must be distinct.
        # Let's just iterate x from min_next to best_last.
        
        for x in range(min_next, best_last + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & local_used:
                continue
            if new_diffs & forbidden:
                continue
            
            row.append(x)
            local_used |= new_diffs
            
            res = dfs(row, local_used)
            if res is not None:
                return res
            
            row.pop()
            local_used -= new_diffs
            
    # Start DFS
    dfs([0], set())
    
    return best_row

def _sa_optimize(rows, n_rows, k, deadline):
    """
    Optimized Simulated Annealing.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build initial used set
    used = set()
    for row in rows:
        used |= _compute_diffs(row)
        
    current_scope = best_scope
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    
    # Track stagnation
    stagnation = 0
    max_stagnation = 2000
    
    while T > min_T and time.time() < deadline - 5:
        # Decide move type: 60% perturb element, 40% rebuild row
        if random.random() < 0.6:
            # Perturb a single element
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try to move value
            # Bias towards reducing scope if this row has the max
            if row[pos] == best_scope:
                # Try smaller values
                new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                # Try random in range
                new_val = random.randint(1, current_scope + 10)
                
            if new_val == old_val:
                continue
                
            # Check order
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
                
            # Compute old diffs involving pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.add(row[j] - old_val)
                
            # Compute new diffs involving pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, len(row)):
                new_diffs.add(row[j] - new_val)
                
            # Check conflict
            remaining = used - old_diffs
            if new_diffs & remaining:
                T *= cooling
                continue
                
            # Valid move
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            
            new_scope = _get_scope(rows)
            delta = new_scope - current_scope
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                # Revert
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs
                stagnation += 1
                
        else:
            # Rebuild a row
            # Pick the row with the largest max element, or random
            if random.random() < 0.5:
                # Pick row with max element
                max_row_idx = 0
                max_val = -1
                for i, r in enumerate(rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_row_idx = i
                r_idx = max_row_idx
            else:
                r_idx = random.randint(0, n_rows - 1)
                
            old_row = rows[r_idx]
            old_diffs = _compute_diffs(old_row)
            
            # Remove old row's diffs from global used
            used -= old_diffs
            
            # Try to rebuild with DFS
            # Give it a hint: current best scope
            new_row = _rebuild_row_dfs(used, k, best_scope, deadline - 1)
            
            if new_row is not None:
                new_diffs = _compute_diffs(new_row)
                
                # Check if new diffs conflict with current used (which excludes old row)
                if not (new_diffs & used):
                    rows[r_idx] = new_row
                    used |= new_diffs
                    
                    new_scope = _get_scope(rows)
                    delta = new_scope - current_scope
                    
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in rows]
                            stagnation = 0
                        else:
                            stagnation += 1
                    else:
                        # Revert
                        rows[r_idx] = old_row
                        used = (used - new_diffs) | old_diffs
                        stagnation += 1
                else:
                    # Conflict, revert
                    used |= old_diffs
                    stagnation += 1
            else:
                # Failed to rebuild, restore old row
                used |= old_diffs
                stagnation += 1
                
        T *= cooling
        
        # If stagnating, try a restart or increase temperature
        if stagnation > max_stagnation:
            # Restart with a new greedy or perturb best
            if time.time() < deadline - 10:
                # Try a new greedy init with some randomness
                new_rows, new_used = _greedy_init(n_rows, k, skip_prob=0.2)
                if _is_valid(new_rows):
                    new_scope = _get_scope(new_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in new_rows]
                        rows = [list(r) for r in new_rows]
                        used = new_used
                        current_scope = new_scope
                        T = 50.0
                        stagnation = 0
                    else:
                        # If new greedy is worse, maybe just perturb current best
                        # For now, just reset stagnation to avoid infinite loop
                        stagnation = 0
            else:
                break
                
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    
    # Try multiple initialization strategies
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Pure greedy
    rows1, _ = _greedy_init(n_rows, k, skip_prob=0.0)
    if _is_valid(rows1):
        scope1 = _get_scope(rows1)
        if scope1 < best_scope:
            best_scope = scope1
            best_rows = [list(r) for r in rows1]
    
    # Strategy 2: Randomized greedy
    for attempt in range(5):
        rows2, _ = _greedy_init(n_rows, k, skip_prob=0.1 + 0.05 * attempt)
        if _is_valid(rows2):
            scope2 = _get_scope(rows2)
            if scope2 < best_scope:
                best_scope = scope2
                best_rows = [list(r) for r in rows2]
    
    if best_rows is None:
        # Fallback to pure greedy
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0)
        return rows
        
    # Optimize
    rows = _sa_optimize(best_rows, n_rows, k, deadline)
    
    # Final check
    if not _is_valid(rows):
        return best_rows
        
    return rows
