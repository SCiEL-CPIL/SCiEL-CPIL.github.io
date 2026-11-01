"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline: 140 seconds (10s safety margin from 150s limit)
    deadline = time.time() + 140.0
    
    # 1. Generate a baseline greedy solution to have a valid fallback
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_score = _max_entry(baseline)
    
    # If the problem is very small, return baseline immediately
    if n_rows <= 2 and k <= 3:
        return baseline
    
    # 2. Local Search / Constraint Satisfaction
    # Strategy:
    # - Initialize with random valid rows (small max entries)
    # - Iteratively repair conflicts and minimize max entry
    # - Use simulated annealing or simple hill climbing with restarts
    
    for attempt in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random valid rows
        rows = _initialize_random_rows(n_rows, k)
        if rows is None:
            continue
        
        # Local search: try to reduce max entry by swapping elements
        current_rows, current_score = _local_search(rows, n_rows, k, deadline)
        
        if current_score < best_score:
            best_score = current_score
            best_rows = current_rows
            if best_score <= 110:
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

def _initialize_random_rows(n_rows, k):
    """
    Initialize n_rows rows with random valid sequences starting at 0.
    Each row has k+1 increasing integers.
    We try to keep them small to allow for a tight final packing.
    """
    rows = []
    for _ in range(n_rows):
        row = _generate_random_row(k)
        if row is None:
            return None
        rows.append(row)
    return rows

def _generate_random_row(k):
    """
    Generate a single random valid row of k+1 increasing integers starting at 0.
    The row must have distinct internal differences.
    We keep the max entry small (e.g., <= 20) to leave room for global packing.
    """
    max_val = 20
    for _ in range(100):
        row = [0]
        for _ in range(k):
            # Pick a random next value larger than the last
            last = row[-1]
            candidates = list(range(last + 1, max_val + 1))
            if not candidates:
                break
            x = random.choice(candidates)
            # Check if adding x creates duplicate differences within this row
            new_diffs = {x - a for a in row}
            existing_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    existing_diffs.add(row[j] - row[i])
            if new_diffs & existing_diffs:
                continue
            row.append(x)
        if len(row) == k + 1:
            return row
    return None

def _local_search(rows, n_rows, k, deadline):
    """
    Local search to minimize the max entry while maintaining validity.
    We use a simple hill-climbing with random restarts of the local moves.
    """
    current_rows = [row[:] for row in rows]
    current_score = _max_entry(current_rows)
    
    # Precompute differences for each row
    row_diffs = [_get_diffs(row) for row in current_rows]
    
    # Local search loop
    for step in range(10000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random row and a random position to try to reduce
        row_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k)  # Positions 1..k (index 0 is fixed at 0)
        
        # Try to reduce the value at pos_idx
        old_val = current_rows[row_idx][pos_idx]
        
        # Try smaller values
        for new_val in range(old_val - 1, 0, -1):
            # Check if new_val > previous element
            if pos_idx > 0 and new_val <= current_rows[row_idx][pos_idx - 1]:
                continue
            # Check if new_val < next element
            if pos_idx < k and new_val >= current_rows[row_idx][pos_idx + 1]:
                continue
            
            # Check if the new row is internally valid
            test_row = current_rows[row_idx][:]
            test_row[pos_idx] = new_val
            if not _is_row_valid(test_row):
                continue
            
            # Check if the new row conflicts with other rows
            new_diffs = _get_diffs(test_row)
            conflict = False
            for other_idx in range(n_rows):
                if other_idx == row_idx:
                    continue
                if new_diffs & row_diffs[other_idx]:
                    conflict = True
                    break
            
            if not conflict:
                # Accept the move
                current_rows[row_idx][pos_idx] = new_val
                row_diffs[row_idx] = new_diffs
                new_score = _max_entry(current_rows)
                if new_score < current_score:
                    current_score = new_score
                break
    
    return current_rows, current_score

def _get_diffs(row):
    """Get the set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_row_valid(row):
    """Check if a single row is valid (increasing, starts at 0, distinct internal diffs)."""
    if len(row) < 2 or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    # Check distinct differences
    diffs = _get_diffs(row)
    expected = len(row) * (len(row) - 1) // 2
    return len(diffs) == expected

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

def _max_entry(rows):
    return max(max(row) for row in rows)
# EVOLVE-BLOCK-END
