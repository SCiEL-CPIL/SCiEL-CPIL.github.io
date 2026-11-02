import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _greedy_build_row(k, freq, target_max):
    """
    Build a single row of k+1 increasing integers starting at 0,
    trying to keep the max entry <= target_max.
    Returns the row or None if impossible.
    """
    n = k + 1
    row = [0]
    used_in_row = set()
    
    for pos in range(1, n):
        # Determine the range of valid values
        lo = row[-1] + 1
        hi = min(target_max, lo + (target_max - lo))
        
        # Try values in order, prefer smaller to keep row compact
        found = False
        for val in range(lo, hi + 1):
            # Check all differences with previous elements
            valid = True
            new_diffs = []
            for prev in row:
                d = val - prev
                if d in used_in_row:
                    valid = False
                    break
                if freq[d] > 0:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(val)
                used_in_row.update(new_diffs)
                for d in new_diffs:
                    freq[d] += 1
                found = True
                break
        
        if not found:
            # Roll back
            for prev in row:
                pass  # We need to undo the freq updates
            # Actually we need to be more careful - let's redo this
            return None
    
    return row

def _greedy_build_row_v2(k, freq, target_max):
    """
    Build a single row, tracking freq updates so we can roll back.
    """
    n = k + 1
    row = [0]
    used_in_row = set()
    freq_changes = []  # list of (diff, delta) for rollback
    
    for pos in range(1, n):
        lo = row[-1] + 1
        hi = target_max
        
        found = False
        for val in range(lo, hi + 1):
            valid = True
            new_diffs = []
            for prev in row:
                d = val - prev
                if d in used_in_row:
                    valid = False
                    break
                if freq[d] > 0:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                row.append(val)
                used_in_row.update(new_diffs)
                for d in new_diffs:
                    freq[d] += 1
                    freq_changes.append((d, 1))
                found = True
                break
        
        if not found:
            # Roll back all changes
            for d, delta in freq_changes:
                freq[d] -= delta
            return None
    
    return row, freq_changes

def _rollback_freq(freq, changes):
    for d, delta in changes:
        freq[d] -= delta

def _row_diffs(row):
    """Return list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _try_rebuild_row(row, k, freq, target_max):
    """
    Try to rebuild a row to have max <= target_max.
    Returns (new_row, freq_changes) or (None, None).
    """
    # First, remove this row's diffs from freq
    old_diffs = _row_diffs(row)
    for d in old_diffs:
        freq[d] -= 1
    
    # Try to build a new row
    result = _greedy_build_row_v2(k, freq, target_max)
    
    if result is None:
        # Restore old diffs
        for d in old_diffs:
            freq[d] += 1
        return None, None
    
    new_row, freq_changes = result
    return new_row, freq_changes

def _local_search_row(row, k, freq, target_max):
    """
    Try to reduce a row's max entry by local search.
    Returns (new_row, success).
    """
    current = row[:]
    current_max = current[k]
    
    if current_max <= target_max:
        return current, True
    
    # Try to reduce the last element first
    best = current[:]
    best_max = current_max
    
    # Remove current row's diffs
    old_diffs = _row_diffs(current)
    for d in old_diffs:
        freq[d] -= 1
    
    # Try to rebuild with lower target
    for t in range(current_max - 1, max(target_max, 0), -1):
        result = _greedy_build_row_v2(k, freq, t)
        if result is not None:
            new_row, _ = result
            best = new_row
            best_max = t
            break
    
    # If we found a better row, update freq
    if best_max < current_max:
        new_diffs = _row_diffs(best)
        for d in new_diffs:
            freq[d] += 1
        return best, True
    else:
        # Restore
        for d in old_diffs:
            freq[d] += 1
        return current, False

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Frequency array - need to cover up to a reasonable max
    # Initial estimate: greedy gives about n_rows * k * something
    # Start with a generous bound
    max_bound = 200
    freq = np.zeros(max_bound + 1, dtype=np.int32)
    
    # Build initial rows greedily
    rows = []
    for r in range(n_rows):
        # Try to build row with increasing target
        row = None
        for t in range(20, max_bound + 1):
            result = _greedy_build_row_v2(k, freq, t)
            if result is not None:
                row, _ = result
                break
        if row is None:
            # Shouldn't happen, but fallback
            row = [0]
            for i in range(1, k + 1):
                row.append(row[-1] + 1)
            # Add diffs
            for d in _row_diffs(row):
                if d < max_bound + 1:
                    freq[d] += 1
        
        rows.append(row)
    
    # Compute current global max
    current_max = max(max(r) for r in rows)
    best_max = current_max
    best_rows = [r[:] for r in rows]
    
    # Phase 1: Try to reduce each row's max
    for r in range(n_rows):
        if time.time() > deadline - 30:
            break
        new_row, success = _local_search_row(rows[r], k, freq, 105)
        if success and max(new_row) < max(rows[r]):
            rows[r] = new_row
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
    
    # Phase 2: Iterative improvement - try to reduce the max row
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        # Find the row with the maximum entry
        max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        current_row = rows[max_row_idx]
        row_max = max(current_row)
        
        if row_max <= best_max:
            # All rows are at or below best
            # Try random rows
            ri = random.randrange(n_rows)
        else:
            ri = max_row_idx
        
        row = rows[ri]
        row_max = max(row)
        
        # Try to reduce this row
        # Remove its diffs
        old_diffs = _row_diffs(row)
        for d in old_diffs:
            if d < len(freq):
                freq[d] -= 1
        
        # Try to rebuild with lower target
        target = row_max - 1
        improved = False
        for t in range(target, max(105, row_max - 20), -1):
            result = _greedy_build_row_v2(k, freq, t)
            if result is not None:
                new_row, _ = result
                new_diffs = _row_diffs(new_row)
                for d in new_diffs:
                    freq[d] += 1
                rows[ri] = new_row
                improved = True
                no_improve = 0
                current_max = max(max(r) for r in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
                break
        
        if not improved:
            # Restore old diffs
            for d in old_diffs:
                if d < len(freq):
                    freq[d] += 1
            no_improve += 1
        
        # Occasionally try a random row
        if random.random() < 0.2:
            ri2 = random.randrange(n_rows)
            row2 = rows[ri2]
            row2_max = max(row2)
            
            old_diffs2 = _row_diffs(row2)
            for d in old_diffs2:
                if d < len(freq):
                    freq[d] -= 1
            
            improved2 = False
            for t in range(row2_max - 1, max(105, row2_max - 15), -1):
                result = _greedy_build_row_v2(k, freq, t)
                if result is not None:
                    new_row2, _ = result
                    new_diffs2 = _row_diffs(new_row2)
                    for d in new_diffs2:
                        freq[d] += 1
                    rows[ri2] = new_row2
                    improved2 = True
                    no_improve = 0
                    current_max = max(max(r) for r in rows)
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
                    break
            
            if not improved2:
                for d in old_diffs2:
                    if d < len(freq):
                        freq[d] += 1
                no_improve += 1
    
    # Phase 3: Final polish - try to squeeze out more
    if time.time() < deadline - 5:
        for r in range(n_rows):
            if time.time() > deadline - 2:
                break
            row = rows[r]
            row_max = max(row)
            
            old_diffs = _row_diffs(row)
            for d in old_diffs:
                if d < len(freq):
                    freq[d] -= 1
            
            improved = False
            for t in range(row_max - 1, max(105, row_max - 10), -1):
                result = _greedy_build_row_v2(k, freq, t)
                if result is not None:
                    new_row, _ = result
                    new_diffs = _row_diffs(new_row)
                    for d in new_diffs:
                        freq[d] += 1
                    rows[r] = new_row
                    improved = True
                    break
            
            if not improved:
                for d in old_diffs:
                    if d < len(freq):
                        freq[d] += 1
    
    # Return the best found
    return best_rows

# EVOLVE-BLOCK-END
