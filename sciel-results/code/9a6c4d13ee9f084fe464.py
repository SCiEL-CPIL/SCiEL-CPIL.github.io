"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct
    across all rows. Minimize the largest entry (scope).
    
    Uses a fast randomized local search:
    1. Greedy initialization.
    2. Iteratively try to reduce the scope by reconstructing rows that currently
       have the maximum value using a fast randomized backtracker.
    3. If stuck, perform random perturbations to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin before 150s kill
    
    # 1. Greedy Initialization
    rows, used_diffs = _greedy_construct(n_rows, k)
    current_scope = _get_scope(rows)
    best_rows = [row[:] for row in rows]
    best_scope = current_scope
    
    rng = random.Random(42)
    
    # 2. Iterative Improvement Loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reset counter and maybe perturb harder or restart local search
            no_improve_count = 0
            # Try to shuffle a random row to escape
            idx = rng.randint(0, n_rows - 1)
            row_to_replace = rows[idx]
            row_diffs = _get_diffs(row_to_replace)
            used_diffs -= row_diffs
            
            # Try to reconstruct with current scope
            new_row = _fast_random_row(k, current_scope, used_diffs, deadline, rng)
            
            if new_row is not None and new_row != row_to_replace:
                rows[idx] = new_row
                used_diffs |= _get_diffs(new_row)
            else:
                used_diffs |= row_diffs
            continue

        # Identify rows contributing to the current scope
        max_rows = [i for i, r in enumerate(rows) if max(r) == current_scope]
        
        if not max_rows:
            break
        
        # Try to reduce the scope by one
        target_scope = current_scope - 1
        
        if target_scope < k:
            break
            
        success = False
        attempts = 0
        max_attempts = 20  # Increase attempts as the row constructor is faster
        
        while not success and attempts < max_attempts and time.time() < deadline:
            idx = rng.choice(max_rows)
            row_to_replace = rows[idx]
            
            # Remove the row's diffs from used_diffs
            row_diffs = _get_diffs(row_to_replace)
            used_diffs -= row_diffs
            
            # Try to construct a new row with max <= target_scope
            new_row = _fast_random_row(k, target_scope, used_diffs, deadline, rng)
            
            if new_row is not None:
                # Success: Update rows and used_diffs
                rows[idx] = new_row
                used_diffs |= _get_diffs(new_row)
                current_scope = target_scope
                best_rows = [row[:] for row in rows]
                best_scope = current_scope
                success = True
                no_improve_count = 0
            else:
                # Failure: Revert
                used_diffs |= row_diffs
                attempts += 1
        
        if success:
            continue
        
        # If we couldn't reduce scope, count as no improvement
        no_improve_count += 1

    return best_rows


def _get_scope(rows):
    """Get the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _get_diffs(row):
    """Get the set of positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows, n_rows, k):
    """Check if the rows form a valid difference triangle set."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff = row[j] - row[i]
                if diff <= 0:
                    return False
                if diff in used:
                    return False
                used.add(diff)
    return len(rows) == n_rows


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row with smallest possible marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows, used


def _fast_random_row(k, max_scope, used_diffs, deadline, rng):
    """
    Fast randomized backtracking search to find a row of length k+1 starting at 0,
    strictly increasing, max <= max_scope, with no diffs in used_diffs.
    Returns the row or None.
    Uses random ordering and a strict time limit per call.
    """
    # Time limit for this specific row construction
    row_deadline = time.time() + 0.05  # 50ms per row attempt
    
    def backtrack(current_row, local_diffs):
        if time.time() > row_deadline:
            return None
        
        if len(current_row) == k + 1:
            return current_row[:]
            
        last_val = current_row[-1]
        remaining = k + 1 - len(current_row)
        
        if last_val + remaining > max_scope:
            return None
            
        start_candidate = last_val + 1
        end_candidate = max_scope - (remaining - 1)
        
        if start_candidate > end_candidate:
            return None
            
        # Generate candidates and shuffle them for randomization
        candidates = list(range(start_candidate, end_candidate + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if not valid:
                continue
                
            current_row.append(x)
            local_diffs |= new_diffs
            
            res = backtrack(current_row, local_diffs)
            if res is not None:
                return res
                
            current_row.pop()
            local_diffs -= new_diffs
            
        return None
        
    return backtrack([0], set())


# EVOLVE-BLOCK-END
