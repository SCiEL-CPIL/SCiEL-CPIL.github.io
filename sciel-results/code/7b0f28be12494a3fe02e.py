"""Difference triangle set (4,7) - optimized constructive search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _validate(rows, n_rows, k):
    """Check that all positive differences are distinct across all rows."""
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

def _get_scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_build(n_rows, k, deadline):
    """Original greedy construction as baseline."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if time.time() > deadline:
                break
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        if len(row) < k + 1:
            break
        rows.append(row)
    if len(rows) < n_rows:
        return None
    return rows

def _backtrack_row(row, used, k, deadline, max_scope):
    """Backtracking to complete a single row given current state."""
    if len(row) == k + 1:
        return row
    if time.time() > deadline:
        return None
    
    last = row[-1]
    # Try candidates starting from last+1 up to max_scope
    for x in range(last + 1, max_scope + 1):
        new_diffs = {x - a for a in row}
        if new_diffs & used:
            continue
        row.append(x)
        used.update(new_diffs)
        result = _backtrack_row(row, used, k, deadline, max_scope)
        if result is not None:
            return result
        row.pop()
        used.difference_update(new_diffs)
    return None

def _construct_with_backtrack(n_rows, k, deadline, max_scope=None):
    """Build rows one at a time using backtracking within a scope bound."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        row = [0]
        if max_scope is None:
            # Estimate: need at least k*(k+1)/2 differences per row, but shared
            # Use a generous bound
            est = k * (k + 1) // 2 * (r + 1) + k * k
            row_max = est
        else:
            row_max = max_scope
        
        result = _backtrack_row(row, used, k, deadline, row_max)
        if result is None:
            return None
        rows.append(result)
    
    return rows

def _local_search(rows, n_rows, k, deadline):
    """Try to reduce the scope by local search: move individual elements."""
    if rows is None:
        return None
    
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(best_rows)
    
    def _is_valid_with_scope(rows_, scope_limit):
        used = set()
        for row in rows_:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                if row[i] > scope_limit:
                    return False
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try moving each non-zero element to a smaller valid position
        for r in range(n_rows):
            for pos in range(1, k + 1):
                orig_val = best_rows[r][pos]
                # Try smaller values
                for new_val in range(orig_val - 1, 0, -1):
                    if new_val <= best_rows[r][pos - 1]:
                        break
                    # Check if this move is valid
                    candidate = list(best_rows[r])
                    candidate[pos] = new_val
                    # Check strict increase
                    if pos < k and new_val >= candidate[pos + 1]:
                        continue
                    if _is_valid_with_scope(best_rows, best_scope - 1 if best_scope > 0 else 0):
                        # This is a full revalidation which is expensive
                        pass
                    # Quick check: just validate the modified row's diffs don't conflict
                    # and row is still increasing
                    row_ok = all(candidate[i] < candidate[i+1] for i in range(k))
                    if not row_ok:
                        continue
                    # Full validation
                    test_rows = [list(row) if i != r else candidate for i, row in enumerate(best_rows)]
                    if _is_valid_with_scope(test_rows, best_scope - 1):
                        best_rows = test_rows
                        best_scope -= 1
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set."""
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with greedy baseline
    baseline = _greedy_build(n_rows, k, deadline)
    if baseline is None:
        # Fallback: return something valid even if not optimal
        # Use a simple construction with large enough scope
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
    
    best_rows = [list(r) for r in baseline]
    best_scope = _get_scope(best_rows)
    
    # Try to improve with local search
    if time.time() < deadline:
        improved = _local_search(best_rows, n_rows, k, deadline)
        if improved is not None:
            best_rows = improved
            best_scope = _get_scope(best_rows)
    
    # Try randomized restarts with different orderings
    if time.time() < deadline:
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Shuffle the order of rows to try different greedy paths
            # Actually, let's try a different approach: randomized greedy
            used = set()
            rows = []
            success = True
            for r in range(n_rows):
                row = [0]
                x = 0
                max_x = best_scope  # Don't exceed current best
                while len(row) < k + 1:
                    x += 1
                    if x > max_x:
                        success = False
                        break
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                if not success:
                    break
                rows.append(row)
            
            if success and len(rows) == n_rows:
                scope = _get_scope(rows)
                if scope < best_scope:
                    best_rows = rows
                    best_scope = scope
                    # Continue trying to improve
                    if time.time() < deadline:
                        improved = _local_search(best_rows, n_rows, k, deadline)
                        if improved is not None:
                            best_rows = improved
                            best_scope = _get_scope(best_rows)
    
    # Final validation
    if not _validate(best_rows, n_rows, k):
        # Return baseline if something went wrong
        return baseline
    
    return best_rows

# EVOLVE-BLOCK-END
