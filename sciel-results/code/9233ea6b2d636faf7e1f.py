import time
import random
import math
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

def _greedy_init(n_rows, k, skip_prob=0.0, seed_offset=0):
    """
    Greedy initialization with optional random skipping.
    """
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if skip_prob > 0 and rng.random() < skip_prob:
                    # Skip some valid values for diversity
                    skipped = False
                    for extra in range(rng.randint(1, 5)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        row.append(x)
                        used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _rebuild_row_dfs(forbidden, k, max_scope, rng):
    """
    Rebuild a row using DFS with pruning.
    Returns a valid row or None.
    """
    row = [0]
    
    def dfs(pos, current_max):
        if pos == k + 1:
            return True
        # Try values in a randomized order biased toward smaller values
        start = row[-1] + 1
        end = min(max_scope, current_max + 20)
        candidates = list(range(start, end + 1))
        rng.shuffle(candidates)
        # Bias: try smaller values first with some probability
        if rng.random() < 0.5:
            candidates.sort()
        
        for val in candidates:
            if val > max_scope:
                continue
            new_diffs = {val - a for a in row}
            if new_diffs & forbidden:
                continue
            # Check internal consistency
            if len(new_diffs) != len(row):
                continue
            row.append(val)
            if dfs(pos + 1, max(current_max, val)):
                return True
            row.pop()
        return False
    
    if dfs(0, 0):
        return list(row)
    return None

def _sa_optimize(rows, n_rows, k, deadline):
    """
    Simulated Annealing with element perturbations and row rebuilds.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build initial used set
    used = set()
    for row in rows:
        used |= _compute_diffs(row)
    
    current_scope = best_scope
    T = 80.0
    cooling = 0.9985
    min_T = 0.3
    
    rng = random.Random()
    
    while T > min_T and time.time() < deadline:
        # Decide move type
        r = rng.random()
        if r < 0.65:
            # Perturb a single element
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Generate candidate new value
            if rng.random() < 0.5:
                # Try to reduce
                new_val = rng.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                # Try random nearby
                delta = rng.randint(-15, 15)
                new_val = max(1, old_val + delta)
                new_val = min(new_val, current_scope + 10)
            
            if new_val == old_val:
                T *= cooling
                continue
            
            # Check ordering
            if pos > 0 and new_val <= row[pos - 1]:
                T *= cooling
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                T *= cooling
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
            
            # Check conflict with remaining used diffs
            remaining = used - old_diffs
            if new_diffs & remaining:
                T *= cooling
                continue
            
            # Valid move - apply
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            
            new_scope = _get_scope(rows)
            delta = new_scope - current_scope
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Revert
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs
                
        elif r < 0.90:
            # Rebuild a row using DFS
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = _compute_diffs(old_row)
            
            # Remove old row's diffs
            used -= old_diffs
            
            # Try to rebuild with slightly relaxed scope
            max_scope = best_scope + rng.randint(0, 15)
            new_row = _rebuild_row_dfs(used, k, max_scope, rng)
            
            if new_row is not None:
                new_diffs = _compute_diffs(new_row)
                if not (new_diffs & used):
                    rows[r_idx] = new_row
                    used |= new_diffs
                    
                    new_scope = _get_scope(rows)
                    delta = new_scope - current_scope
                    
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in rows]
                    else:
                        # Revert
                        rows[r_idx] = old_row
                        used = (used - new_diffs) | old_diffs
                else:
                    used |= old_diffs
            else:
                used |= old_diffs
        else:
            # Compress: try to shift a row down
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            old_diffs = _compute_diffs(row)
            used -= old_diffs
            
            # Try to find a smaller valid row by shifting
            best_new_row = None
            for shift in range(1, min(10, row[-1] - k)):
                candidate = [0] + [v - shift for v in row[1:] if v > shift]
                if len(candidate) < k + 1:
                    continue
                candidate = candidate[:k+1]
                # Check if increasing
                valid = True
                for i in range(1, len(candidate)):
                    if candidate[i] <= candidate[i-1]:
                        valid = False
                        break
                if not valid:
                    continue
                cd = _compute_diffs(candidate)
                if cd & used:
                    continue
                if candidate[-1] < row[-1]:
                    best_new_row = candidate
                    break
            
            if best_new_row is not None:
                new_diffs = _compute_diffs(best_new_row)
                if not (new_diffs & used):
                    rows[r_idx] = best_new_row
                    used |= new_diffs
                    new_scope = _get_scope(rows)
                    delta = new_scope - current_scope
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in rows]
                    else:
                        rows[r_idx] = old_row
                        used = (used - new_diffs) | old_diffs
                else:
                    used |= old_diffs
            else:
                used |= old_diffs
        
        T *= cooling
        
        # Periodic restart if stuck
        if T < 15.0 and rng.random() < 0.02 and time.time() < deadline - 30:
            # Try a fresh greedy init
            skip = rng.random() * 0.15
            new_rows, new_used = _greedy_init(n_rows, k, skip_prob=skip, seed_offset=rng.randint(0, 10000))
            if _is_valid(new_rows):
                new_scope = _get_scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = [list(r) for r in new_rows]
                    used = new_used
                    current_scope = new_scope
                    T = 40.0
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 130.0  # Safety margin (150s limit)
    
    # Multi-start greedy initialization
    best_rows = None
    best_scope = float('inf')
    
    # Try several different initializations
    for i in range(8):
        skip = 0.0 if i == 0 else 0.03 + 0.02 * i
        rows, _ = _greedy_init(n_rows, k, skip_prob=skip, seed_offset=i * 100)
        if _is_valid(rows):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0, seed_offset=0)
        return rows
    
    # Optimize with SA
    rows = _sa_optimize(best_rows, n_rows, k, deadline)
    
    # Final validation
    if not _is_valid(rows):
        return best_rows
    
    return rows
