"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
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
    
    if n_rows <= 2 and k <= 3:
        return baseline
    
    # 2. Simulated Annealing
    # Strategy:
    # - Start with a random valid configuration (if possible) or the greedy one
    # - Use a cost function that penalizes both max entry and conflicts
    # - Allow "bad" moves (increasing cost) with a probability based on temperature
    # - Cool down over time
    
    # Pre-generate a pool of valid rows to use for "replace" moves
    # This is crucial for speed: we don't want to generate rows on the fly during SA
    row_pool = []
    for _ in range(200):
        row = _generate_random_row(k, max_val=150)
        if row:
            row_pool.append(row)
            
    # If pool is small, add more
    while len(row_pool) < 50:
        row = _generate_random_row(k, max_val=150)
        if row:
            row_pool.append(row)

    # Initial state: use the greedy baseline as it's valid and likely better than random
    current_rows = [row[:] for row in baseline]
    current_diffs = [_get_diffs(row) for row in current_rows]
    current_cost = _calculate_cost(current_rows, current_diffs)
    
    best_rows_sa = [row[:] for row in current_rows]
    best_cost = current_cost
    best_score_sa = best_score
    
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    while T > T_min and time.time() < deadline - 5:
        # Move selection
        move_type = random.random()
        
        if move_type < 0.3:
            # Swap two rows
            if n_rows < 2: continue
            i, j = random.sample(range(n_rows), 2)
            current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
            current_diffs[i], current_diffs[j] = current_diffs[j], current_diffs[i]
            new_cost = _calculate_cost(current_rows, current_diffs)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows_sa = [row[:] for row in current_rows]
                    new_score = _max_entry(best_rows_sa)
                    if new_score < best_score_sa:
                        best_score_sa = new_score
            else:
                # Undo swap
                current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
                current_diffs[i], current_diffs[j] = current_diffs[j], current_diffs[i]
                # Accept with probability
                if random.random() < _accept_probability(current_cost, new_cost, T):
                    current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
                    current_diffs[i], current_diffs[j] = current_diffs[j], current_diffs[i]
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows_sa = [row[:] for row in current_rows]
                        new_score = _max_entry(best_rows_sa)
                        if new_score < best_score_sa:
                            best_score_sa = new_score
                else:
                    # Reject
                    pass
        elif move_type < 0.6:
            # Shift a row (change one element)
            row_idx = random.randint(0, n_rows - 1)
            pos_idx = random.randint(1, k)
            old_val = current_rows[row_idx][pos_idx]
            
            # Try a random new value
            # Range: between prev and next, or slightly outside to allow rearrangement?
            # Strictly inside is safer for validity, but SA needs flexibility.
            # Let's try shifting by a small delta first, then random.
            if random.random() < 0.5:
                delta = random.randint(-10, 10)
                new_val = old_val + delta
            else:
                low = current_rows[row_idx][pos_idx - 1] + 1
                high = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val + 20
                if low > high:
                    continue
                new_val = random.randint(low, high)
            
            if new_val <= current_rows[row_idx][pos_idx - 1] or (pos_idx < k and new_val >= current_rows[row_idx][pos_idx + 1]):
                continue
                
            test_row = current_rows[row_idx][:]
            test_row[pos_idx] = new_val
            
            # Check internal validity of the row
            if not _is_row_valid_fast(test_row):
                continue
                
            new_diffs = _get_diffs(test_row)
            
            # Calculate cost delta
            # We need to know which diffs are removed and added
            old_diffs = current_diffs[row_idx]
            
            # Remove old diffs from global view (conceptually)
            # Add new diffs
            # For cost calculation, we just recompute the whole cost for simplicity in the prototype,
            # but for speed we should optimize. Given 150s, full recompute of cost is O(N*K^2).
            # N=7, K=6. 7 * 21 = 147 ops. Very fast.
            
            # Temporarily update
            current_rows[row_idx] = test_row
            current_diffs[row_idx] = new_diffs
            
            new_cost = _calculate_cost(current_rows, current_diffs)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows_sa = [row[:] for row in current_rows]
                    new_score = _max_entry(best_rows_sa)
                    if new_score < best_score_sa:
                        best_score_sa = new_score
            else:
                # Undo
                current_rows[row_idx] = current_rows[row_idx][:] # Restore old row? 
                # Wait, I modified current_rows in place. I need to restore.
                current_rows[row_idx][pos_idx] = old_val
                current_diffs[row_idx] = old_diffs
                
                if random.random() < _accept_probability(current_cost, new_cost, T):
                    current_rows[row_idx][pos_idx] = new_val
                    current_diffs[row_idx] = new_diffs
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows_sa = [row[:] for row in current_rows]
                        new_score = _max_entry(best_rows_sa)
                        if new_score < best_score_sa:
                            best_score_sa = new_score
        
        else:
            # Replace a row with a random valid row from pool
            row_idx = random.randint(0, n_rows - 1)
            # Pick a random row from pool that is different
            new_row = random.choice(row_pool)
            
            # Check if it's identical
            if current_rows[row_idx] == new_row:
                continue
                
            test_row = new_row[:]
            new_diffs = _get_diffs(test_row)
            
            # Temporarily update
            old_row = current_rows[row_idx][:]
            old_diffs = current_diffs[row_idx]
            
            current_rows[row_idx] = test_row
            current_diffs[row_idx] = new_diffs
            
            new_cost = _calculate_cost(current_rows, current_diffs)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows_sa = [row[:] for row in current_rows]
                    new_score = _max_entry(best_rows_sa)
                    if new_score < best_score_sa:
                        best_score_sa = new_score
            else:
                # Undo
                current_rows[row_idx] = old_row
                current_diffs[row_idx] = old_diffs
                
                if random.random() < _accept_probability(current_cost, new_cost, T):
                    current_rows[row_idx] = test_row
                    current_diffs[row_idx] = new_diffs
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows_sa = [row[:] for row in current_rows]
                        new_score = _max_entry(best_rows_sa)
                        if new_score < best_score_sa:
                            best_score_sa = new_score

        T *= cooling

    # Final check: if SA found a valid solution better than baseline, use it
    # Note: SA cost includes conflict penalty. We want to return a VALID solution.
    # If best_rows_sa is valid, use it. Otherwise, stick to baseline or try to repair best_rows_sa.
    if _is_valid(best_rows_sa, n_rows, k):
        sa_score = _max_entry(best_rows_sa)
        if sa_score < best_score:
            best_rows = best_rows_sa
            best_score = sa_score
    
    # If we still have time, do a final local polish on the best valid solution found
    if time.time() < deadline - 5:
        final_rows, final_score = _local_search(best_rows, n_rows, k, deadline)
        if final_score < best_score:
            best_rows = final_rows
            best_score = final_score

    return best_rows

def _accept_probability(current_cost, new_cost, T):
    if new_cost < current_cost:
        return 1.0
    delta = new_cost - current_cost
    if T == 0:
        return 0.0
    return math.exp(-delta / T)

def _calculate_cost(rows, diffs):
    # Cost = Max Entry + Penalty for Conflicts
    # Conflict penalty: number of duplicate differences across all rows
    # We want to minimize max entry primarily, but conflicts must be 0.
    # If we allow conflicts, we need a heavy penalty.
    
    max_entry = max(max(row) for row in rows)
    
    # Count conflicts
    # Flatten all diffs
    all_diffs = []
    for d_set in diffs:
        all_diffs.extend(d_set)
        
    # Count duplicates
    # Using a counter is efficient enough for small N
    counts = {}
    for d in all_diffs:
        counts[d] = counts.get(d, 0) + 1
        
    conflicts = sum(c - 1 for c in counts.values() if c > 1)
    
    # Penalty weight: make conflicts very expensive
    # If max entry is ~100, and we have 1 conflict, we want cost to jump significantly.
    # Let's say 1 conflict = 1000 cost units.
    return max_entry + conflicts * 1000.0

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

def _generate_random_row(k, max_val=100):
    """
    Generate a single valid row of k+1 increasing integers starting at 0.
    """
    for _ in range(50):
        row = [0]
        for _ in range(k):
            last = row[-1]
            candidates = list(range(last + 1, max_val + 1))
            if not candidates:
                break
            x = random.choice(candidates)
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

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_row_valid_fast(row):
    """Check if a single row is valid (increasing, starts at 0, distinct internal diffs)."""
    if len(row) < 2 or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
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

def _local_search(rows, n_rows, k, deadline):
    """
    Final polish: strict hill climbing on a valid solution.
    """
    current_rows = [row[:] for row in rows]
    current_score = _max_entry(current_rows)
    row_diffs = [_get_diffs(row) for row in current_rows]
    
    # Try to reduce max entry by moving elements down
    # Identify the row with the max entry
    max_row_idx = max(range(n_rows), key=lambda i: max(current_rows[i]))
    
    for step in range(1000):
        if time.time() > deadline - 2:
            break
            
        improved = False
        
        # Try to reduce the largest element in the max row
        row = current_rows[max_row_idx]
        max_pos = len(row) - 1
        old_val = row[max_pos]
        
        for new_val in range(old_val - 1, row[max_pos - 1], -1):
            test_row = row[:]
            test_row[max_pos] = new_val
            if not _is_row_valid_fast(test_row):
                continue
            
            new_diffs = _get_diffs(test_row)
            conflict = False
            for other_idx in range(n_rows):
                if other_idx == max_row_idx:
                    continue
                if new_diffs & row_diffs[other_idx]:
                    conflict = True
                    break
            
            if not conflict:
                current_rows[max_row_idx] = test_row
                row_diffs[max_row_idx] = new_diffs
                new_score = _max_entry(current_rows)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Update max row idx if needed
                    max_row_idx = max(range(n_rows), key=lambda i: max(current_rows[i]))
                    break
        
        if not improved:
            # Try other rows or positions
            # Simple: break if no improvement on max row
            break
            
    return current_rows, current_score

# EVOLVE-BLOCK-END
