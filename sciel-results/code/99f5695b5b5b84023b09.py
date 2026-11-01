"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_scope(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check that all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy_row(used, k, start_val=0):
    """Build one row greedily: start at 0, pick smallest next mark with unused diffs."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used.update(new_diffs)
    return row

def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes smallest next marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _local_search(rows, k, max_iters=5000, time_limit=10.0):
    """
    Local search: try to reduce the scope by moving individual elements.
    A move changes one element in one row to a different value, keeping the row
    strictly increasing and all differences globally unique.
    Objective: minimize the maximum entry (scope).
    """
    start_time = time.time()
    n_rows = len(rows)
    best_rows = [row[:] for row in rows]
    best_scope = _compute_scope(best_rows)
    current_rows = [row[:] for row in best_rows]
    current_scope = best_scope
    
    # Precompute used differences
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    used.add(row[j] - row[i])
        return used
    
    def check_move(row, idx, new_val):
        """Check if setting row[idx] = new_val keeps row valid and diffs unused by others."""
        if idx == 0:
            if new_val != 0:
                return False
        else:
            if new_val <= row[idx - 1]:
                return False
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                return False
        
        # Compute diffs that would be removed and added
        old_val = row[idx]
        if new_val == old_val:
            return True, set(), set()
        
        removed = set()
        added = set()
        n = len(row)
        for i in range(n):
            if i == idx:
                continue
            if i < idx:
                old_d = old_val - row[i]
                new_d = new_val - row[i]
                removed.add(old_d)
                added.add(new_d)
            else:
                old_d = row[i] - old_val
                new_d = row[i] - new_val
                removed.add(old_d)
                added.add(new_d)
        
        # Check added diffs don't conflict with any other row's diffs or within same row
        # First check within same row: added diffs must be distinct from each other and from remaining row diffs
        if len(added) != len(removed):
            # Some diffs might collide within the row
            pass
        
        # Check added diffs are not in any other row
        # We'll do a full check for simplicity in the candidate
        return True, removed, added
    
    def full_validate(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    rng = random.Random(42)
    iters = 0
    
    while iters < max_iters and (time.time() - start_time) < time_limit:
        iters += 1
        
        # Pick a random row and position to modify
        r_idx = rng.randrange(n_rows)
        row = current_rows[r_idx]
        pos = rng.randrange(1, len(row))  # don't modify position 0
        
        # Try to reduce: move the element to a smaller value
        old_val = row[pos]
        # Candidate range: (row[pos-1], row[pos+1] if exists)
        lo = row[pos - 1] + 1
        hi = row[pos + 1] - 1 if pos < len(row) - 1 else old_val - 1
        
        if lo > hi:
            continue
        
        # Try a random value in range, biased toward smaller values
        # Focus on reducing scope: if this is the max element, try to reduce it
        target = rng.randint(lo, min(hi, old_val - 1)) if old_val > lo else None
        if target is None or target == old_val:
            # Try a small random perturbation
            if old_val > lo:
                target = rng.randint(lo, old_val - 1)
            else:
                continue
        
        # Test the move
        test_row = row[:]
        test_row[pos] = target
        
        # Check row is still strictly increasing
        for i in range(1, len(test_row)):
            if test_row[i] <= test_row[i-1]:
                break
        else:
            # Check all differences are unique across all rows
            test_rows = [current_rows[i][:] for i in range(n_rows)]
            test_rows[r_idx] = test_row
            if full_validate(test_rows):
                new_scope = _compute_scope(test_rows)
                if new_scope < current_scope:
                    current_rows = test_rows
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [row[:] for row in current_rows]
    
    return best_rows, best_scope

def _il_search(rows, k, n_restarts=5, time_limit=30.0):
    """
    Iterated local search with jittered rebuilds.
    """
    start_time = time.time()
    n_rows = len(rows)
    best_rows = [row[:] for row in rows]
    best_scope = _compute_scope(best_rows)
    
    rng = random.Random(123)
    
    for restart in range(n_restarts):
        if (time.time() - start_time) > time_limit - 5:
            break
        
        # Jitter: rebuild some rows with randomization
        work_rows = [row[:] for row in best_rows]
        
        # Randomly shuffle the order of row construction for diversity
        order = list(range(n_rows))
        rng.shuffle(order)
        
        # Rebuild with greedy but with random offsets
        for r_idx in order:
            if rng.random() < 0.3:  # 30% chance to keep original row
                continue
            # Rebuild this row from scratch, removing its diffs from used set
            used = set()
            for other_idx, other_row in enumerate(work_rows):
                if other_idx == r_idx:
                    continue
                n = len(other_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        used.add(other_row[j] - other_row[i])
            
            # Greedy rebuild with slight randomization
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
            
            work_rows[r_idx] = row
        
        # Local search on the jittered solution
        ls_rows, ls_scope = _local_search(work_rows, k, max_iters=3000, time_limit=15.0)
        
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows
    
    return best_rows, best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry (scope).
    """
    total_deadline = time.time() + 140  # safety margin before 150s kill
    
    # Phase 1: Greedy initial solution
    rows = _greedy_construct(n_rows, k)
    scope = _compute_scope(rows)
    
    # Phase 2: Local search to improve
    rows, scope = _local_search(rows, k, max_iters=10000, time_limit=20.0)
    
    # Phase 3: Iterated local search with jittered rebuilds
    if time.time() < total_deadline - 40:
        rows, scope = _il_search(rows, k, n_restarts=8, time_limit=60.0)
    
    # Phase 4: Final polish - targeted descent on the max-value row
    if time.time() < total_deadline - 10:
        # Find the row with the maximum entry and try to reduce it
        for _ in range(3):
            if time.time() >= total_deadline - 5:
                break
            max_row_idx = 0
            max_val = -1
            for i, row in enumerate(rows):
                if max(row) > max_val:
                    max_val = max(row)
                    max_row_idx = i
            
            # Try to reduce elements in the max row
            row = rows[max_row_idx]
            improved = True
            while improved and time.time() < total_deadline - 5:
                improved = False
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() >= total_deadline - 5:
                        break
                    old_val = row[pos]
                    lo = row[pos - 1] + 1
                    hi = row[pos + 1] - 1 if pos < len(row) - 1 else old_val - 1
                    if lo >= old_val:
                        continue
                    # Try values from old_val-1 down to lo
                    for new_val in range(old_val - 1, lo - 1, -1):
                        test_row = row[:]
                        test_row[pos] = new_val
                        test_rows = [rows[i][:] for i in range(n_rows)]
                        test_rows[max_row_idx] = test_row
                        if _is_valid(test_rows):
                            rows = test_rows
                            improved = True
                            break
                    if improved:
                        break
    
    return rows

# EVOLVE-BLOCK-END
