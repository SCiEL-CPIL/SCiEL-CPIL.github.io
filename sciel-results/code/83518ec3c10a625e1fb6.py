"""
Strategy: Deterministic Greedy Construction + Steepest Descent Repair.
1. Construct rows greedily to ensure a valid, compact baseline.
2. Use a repair loop that targets the global maximum entry.
3. For the row containing the max entry, try to reduce its largest element 
   by the smallest possible integer delta, checking for conflicts.
4. If stuck, try reducing other elements in that row, or shifting the whole row down.
5. This avoids the random walk inefficiency of SA and focuses directly on minimizing the scope.
"""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    
    # 1. Baseline: Greedy Construction
    # This ensures we start with a valid solution.
    best_rows = _greedy_construct(n_rows, k)
    best_score = _max_entry(best_rows)
    
    # If the problem is trivial, return immediately
    if n_rows <= 2 and k <= 3:
        return best_rows

    # 2. Local Search / Repair Phase
    # We will try to improve the baseline by reducing the max entry.
    current_rows = [row[:] for row in best_rows]
    current_score = best_score
    
    # Pre-compute diffs for speed
    current_diffs = [_get_diffs(row) for row in current_rows]
    
    # We use a "stagnation counter" to allow some randomization if deterministic moves fail
    stagnation_count = 0
    max_stagnation = 50
    
    while time.time() < deadline - 2:
        # Identify the row with the maximum entry
        max_val = -1
        max_row_idx = -1
        for i in range(n_rows):
            if current_rows[i][-1] > max_val:
                max_val = current_rows[i][-1]
                max_row_idx = i
        
        if max_row_idx == -1:
            break
            
        improved = False
        
        # Strategy A: Try to reduce the last element of the max row
        row = current_rows[max_row_idx]
        last_idx = len(row) - 1
        old_val = row[last_idx]
        
        # Try reducing the last element by 1, 2, ... up to a limit
        # We check from smallest reduction first (steepest descent towards smaller values)
        # Actually, we want to find the *new* smallest valid value.
        # So we iterate new_val from row[last_idx-1]+1 up to old_val-1.
        # Wait, if we iterate upwards from the minimum, we find the *smallest* valid value.
        # That is the best move.
        
        found_new_val = False
        for new_val in range(row[last_idx - 1] + 1, old_val):
            test_row = row[:]
            test_row[last_idx] = new_val
            
            # Check internal validity
            if not _is_row_valid_fast(test_row):
                continue
                
            new_diffs = _get_diffs(test_row)
            
            # Check conflicts with other rows
            conflict = False
            for j in range(n_rows):
                if j == max_row_idx:
                    continue
                if new_diffs & current_diffs[j]:
                    conflict = True
                    break
            
            if not conflict:
                # Accept move
                current_rows[max_row_idx] = test_row
                current_diffs[max_row_idx] = new_diffs
                current_score = _max_entry(current_rows)
                improved = True
                found_new_val = True
                break
        
        if not found_new_val:
            # Strategy B: Try to reduce other elements in the max row
            # This might allow the last element to be reduced later
            for pos in range(1, last_idx):
                old_v = row[pos]
                # Try to reduce this element
                for new_v in range(row[pos-1]+1, old_v):
                    test_row = row[:]
                    test_row[pos] = new_v
                    if not _is_row_valid_fast(test_row):
                        continue
                    new_diffs = _get_diffs(test_row)
                    conflict = False
                    for j in range(n_rows):
                        if j == max_row_idx:
                            continue
                        if new_diffs & current_diffs[j]:
                            conflict = True
                            break
                    if not conflict:
                        current_rows[max_row_idx] = test_row
                        current_diffs[max_row_idx] = new_diffs
                        # Note: this doesn't necessarily reduce the global max immediately,
                        # but it might unlock a reduction for the last element in the next iteration.
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Strategy C: Randomized Perturbation (Escape Local Minima)
            # If we are stuck, make a random valid change to a non-critical row or position
            # to try to break the deadlock.
            if stagnation_count < max_stagnation:
                stagnation_count += 1
                if _random_perturb(current_rows, current_diffs, n_rows, k, deadline):
                    # Recalculate diffs if changed
                    current_diffs = [_get_diffs(row) for row in current_rows]
                    # Check if score improved (unlikely from random, but possible if we accidentally fixed a previous suboptimal state? 
                    # No, perturb should maintain validity. Score might stay same or increase. 
                    # We only accept if it leads to a better state later? 
                    # For simplicity, we just accept the perturbation to explore.)
                    pass
            else:
                # If we've been stuck for a long time, try a full restart with a different random seed for the greedy part?
                # Or just break.
                break
                
        if improved:
            stagnation_count = 0
            if current_score < best_score:
                best_score = current_score
                best_rows = [row[:] for row in current_rows]
        else:
            stagnation_count += 1
            if stagnation_count > 100:
                break

    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_row_valid_fast(row):
    if len(row) < 2 or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    diffs = _get_diffs(row)
    expected = len(row) * (len(row) - 1) // 2
    return len(diffs) == expected

def _max_entry(rows):
    return max(max(row) for row in rows)

def _random_perturb(rows, diffs, n_rows, k, deadline):
    """
    Attempts a random valid modification to escape local minima.
    Returns True if a modification was made.
    """
    if time.time() > deadline - 1:
        return False
        
    # Pick a random row and position
    row_idx = random.randint(0, n_rows - 1)
    pos_idx = random.randint(1, k)
    
    row = rows[row_idx]
    old_val = row[pos_idx]
    
    # Try a few random new values
    for _ in range(10):
        # Random delta
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Bounds check
        if new_val <= row[pos_idx - 1]:
            continue
        if pos_idx < k and new_val >= row[pos_idx + 1]:
            continue
            
        test_row = row[:]
        test_row[pos_idx] = new_val
        
        if not _is_row_valid_fast(test_row):
            continue
            
        new_diffs = _get_diffs(test_row)
        
        # Check conflicts
        conflict = False
        for j in range(n_rows):
            if j == row_idx:
                continue
            if new_diffs & diffs[j]:
                conflict = True
                break
                
        if not conflict:
            rows[row_idx] = test_row
            diffs[row_idx] = new_diffs
            return True
            
    return False

# Helper for validity check of the whole set (used implicitly in logic)
def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True
