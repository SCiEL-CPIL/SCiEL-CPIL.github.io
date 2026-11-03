"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys
from collections import defaultdict

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_init(n_rows, k):
    """Greedy initialization: build rows one by one with smallest possible marks."""
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

def _rebuild_row(row, used, k, max_val, deadline):
    """
    Try to rebuild a row with entries in [0, max_val] that avoids 'used' differences.
    Returns (new_row, new_diffs) or (None, None).
    Uses backtracking with pruning.
    """
    n = k + 1  # number of elements
    new_row = [0]
    
    def backtrack(pos, current_max):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        # We need to leave room for remaining elements
        if max_val - current_max < (n - pos - 1):
            return False
        start = current_max + 1
        # Upper bound: leave at least (n-pos-1) room
        end = min(max_val, max_val - (n - pos - 1))
        for v in range(start, end + 1):
            # Check new diffs
            ok = True
            for i in range(pos):
                d = v - new_row[i]
                if d in used:
                    ok = False
                    break
            if not ok:
                continue
            new_row.append(v)
            if backtrack(pos + 1, v):
                return True
            new_row.pop()
        return False
    
    if backtrack(1, 0):
        final_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                final_diffs.add(new_row[j] - new_row[i])
        return new_row, final_diffs
    return None, None

def _optimize(rows, n_rows, k, deadline):
    """
    Multi-strategy optimization:
    1. Row reconstruction: rebuild individual rows to fit in smaller scope
    2. Local search with multiple move types
    3. Iterated local search with restarts
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build used set
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    
    # Strategy 1: Aggressive row rebuilding to reduce scope
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Try to reduce the scope by 1
        target_scope = best_scope - 1
        if target_scope < 0:
            break
        
        # Find which row has the max entry
        max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        
        if rows[max_row_idx][-1] <= target_scope:
            continue
        
        old_row = rows[max_row_idx]
        old_diffs = _row_diffs(old_row)
        remaining_used = used - old_diffs
        
        # Rebuild this row
        result = _rebuild_row(old_row, remaining_used, k, target_scope, deadline)
        if result[0] is not None:
            new_row, new_diffs = result
            if not (new_diffs & remaining_used):
                rows[max_row_idx] = new_row
                used = remaining_used | new_diffs
                new_scope = _get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
        
        # Also try rebuilding a random other row to fit in current best_scope
        if random.random() < 0.7:
            r_idx = random.randint(0, n_rows - 1)
            if r_idx == max_row_idx:
                continue
            old_row = rows[r_idx]
            old_diffs = _row_diffs(old_row)
            remaining_used = used - old_diffs
            result = _rebuild_row(old_row, remaining_used, k, best_scope, deadline)
            if result[0] is not None:
                new_row, new_diffs = result
                if not (new_diffs & remaining_used):
                    rows[r_idx] = new_row
                    used = remaining_used | new_diffs
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
    
    # Strategy 2: Simulated annealing with multiple move types
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    
    while T > min_T and time.time() < deadline:
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Single element move
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            max_val = best_scope
            if random.random() < 0.7:
                # Try to decrease
                new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                new_val = random.randint(1, max_val)
            
            if new_val == old_val:
                T *= cooling
                continue
            
            if pos > 0 and new_val <= row[pos - 1]:
                T *= cooling
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                T *= cooling
                continue
            
            # Compute old and new diffs
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.add(row[j] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, len(row)):
                new_diffs.add(row[j] - new_val)
            
            if new_diffs & (used - old_diffs):
                T *= cooling
                continue
            
            # Apply move
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            new_scope = _get_scope(rows)
            delta = new_scope - best_scope
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs
            
        elif move_type < 0.7:
            # Swap two elements in the same row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            if len(row) < 3:
                T *= cooling
                continue
            pos1, pos2 = random.sample(range(1, len(row)), 2)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            
            # Check if swap maintains increasing order
            if pos1 + 1 == pos2:
                # Adjacent swap: need row[pos1-1] < row[pos2] and row[pos1] < row[pos2+1] if exists
                if pos1 > 0 and row[pos1 - 1] >= row[pos2]:
                    T *= cooling
                    continue
                if pos2 < len(row) - 1 and row[pos1] >= row[pos2 + 1]:
                    T *= cooling
                    continue
            else:
                # Non-adjacent swap
                if pos1 > 0 and row[pos1 - 1] >= row[pos2]:
                    T *= cooling
                    continue
                if pos2 < len(row) - 1 and row[pos1] >= row[pos2 + 1]:
                    T *= cooling
                    continue
                # Also check intermediate elements
                for p in range(pos1 + 1, pos2):
                    if row[p] >= row[pos2] or row[p] >= row[pos1]:
                        T *= cooling
                        break
                else:
                    pass
            
            v1, v2 = row[pos1], row[pos2]
            
            # Compute old diffs
            old_diffs = set()
            for i in range(pos1):
                old_diffs.add(v1 - row[i])
            for j in range(pos1 + 1, len(row)):
                if j != pos2:
                    old_diffs.add(row[j] - v1)
            old_diffs.add(v2 - v1)
            for i in range(pos2):
                if i != pos1:
                    old_diffs.add(v2 - row[i])
            for j in range(pos2 + 1, len(row)):
                old_diffs.add(row[j] - v2)
            
            # Compute new diffs after swap
            new_diffs = set()
            for i in range(pos1):
                new_diffs.add(v2 - row[i])
            for j in range(pos1 + 1, len(row)):
                if j != pos2:
                    new_diffs.add(row[j] - v2)
            new_diffs.add(v1 - v2)
            for i in range(pos2):
                if i != pos1:
                    new_diffs.add(v1 - row[i])
            for j in range(pos2 + 1, len(row)):
                new_diffs.add(row[j] - v1)
            
            if new_diffs & (used - old_diffs):
                T *= cooling
                continue
            
            # Apply swap
            row[pos1], row[pos2] = v2, v1
            used = (used - old_diffs) | new_diffs
            new_scope = _get_scope(rows)
            delta = new_scope - best_scope
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                row[pos1], row[pos2] = v1, v2
                used = (used - new_diffs) | old_diffs
        
        else:
            # Move an element to a new position in the same row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            if len(row) < 3:
                T *= cooling
                continue
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try to move to a nearby position
            if random.random() < 0.5:
                new_pos = pos - 1 if pos > 1 else pos + 1
            else:
                new_pos = random.randint(1, len(row) - 1)
            
            if new_pos == pos:
                T *= cooling
                continue
            
            # Check validity
            if new_pos > 0 and old_val <= row[new_pos - 1]:
                T *= cooling
                continue
            if new_pos < len(row) - 1 and old_val >= row[new_pos + 1]:
                T *= cooling
                continue
            
            # Compute old diffs
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.add(row[j] - old_val)
            
            # Simulate the move
            temp_row = row[:]
            temp_row.pop(pos)
            temp_row.insert(new_pos, old_val)
            
            # Check increasing order
            valid = True
            for i in range(len(temp_row) - 1):
                if temp_row[i] >= temp_row[i + 1]:
                    valid = False
                    break
            if not valid:
                T *= cooling
                continue
            
            # Compute new diffs
            new_diffs = set()
            for i in range(new_pos):
                new_diffs.add(old_val - temp_row[i])
            for j in range(new_pos + 1, len(temp_row)):
                new_diffs.add(temp_row[j] - old_val)
            
            if new_diffs & (used - old_diffs):
                T *= cooling
                continue
            
            # Apply move
            old_row_copy = row[:]
            rows[r_idx] = temp_row
            used = (used - old_diffs) | new_diffs
            new_scope = _get_scope(rows)
            delta = new_scope - best_scope
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                rows[r_idx] = old_row_copy
                used = (used - new_diffs) | old_diffs
        
        T *= cooling
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with greedy solution
    rows = _greedy_init(n_rows, k)
    
    # Verify validity
    if not _is_valid(rows):
        return rows
    
    # Optimize
    rows = _optimize(rows, n_rows, k, deadline)
    
    # Final validation
    if not _is_valid(rows):
        return _greedy_init(n_rows, k)
    
    return rows
# EVOLVE-BLOCK-END
