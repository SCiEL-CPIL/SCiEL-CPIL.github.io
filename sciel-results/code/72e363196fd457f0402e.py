"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid(rows):
    """Check if all positive differences within rows are distinct across all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows) if rows else 0

def _greedy_row(used, k, max_val=None, rng=None, skip_prob=0.0):
    """
    Build a single row greedily.
    used: set of already-used differences
    k: number of gaps, so row has k+1 entries starting at 0
    max_val: optional upper bound for the last element
    rng: random.Random instance
    skip_prob: probability to skip a candidate (for randomization)
    """
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 100000
    while len(row) < k + 1:
        x += 1
        if x > limit:
            return None
        if rng is not None and skip_prob > 0:
            # Sometimes skip valid candidates to explore different structures
            if rng.random() < skip_prob:
                continue
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used.update(new_diffs)
    return row

def _local_improve(rows, k, used, max_iter=50, rng=None):
    """
    Try to reduce the scope by locally adjusting entries.
    Right-to-left compaction: try to reduce each element to its minimum feasible value.
    """
    if rng is None:
        rng = random.Random()
    
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for _ in range(max_iter):
        improved = False
        # Work on a copy
        cur_rows = [list(r) for r in best_rows]
        
        # Recompute used set
        cur_used = set()
        for row in cur_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    cur_used.add(row[j] - row[i])
        
        # Right-to-left compaction on each row
        for r_idx in range(len(cur_rows)):
            row = cur_rows[r_idx]
            # Process from right to left (excluding index 0)
            for pos in range(len(row) - 1, 0, -1):
                # Try to reduce row[pos]
                old_val = row[pos]
                # Find minimum feasible value
                # Must be > row[pos-1] and < row[pos+1] if pos < len(row)-1
                lower = row[pos - 1] + 1
                upper = row[pos]  # can try to go lower
                # Check all values from lower to old_val-1
                for new_val in range(lower, old_val):
                    # Check if this value is valid
                    # Remove old diffs involving row[pos]
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    if pos < len(row) - 1:
                        for j in range(pos + 1, len(row)):
                            old_diffs.add(row[j] - old_val)
                    
                    # Tentatively remove old diffs
                    temp_used = cur_used - old_diffs
                    
                    # Add new diffs
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    if pos < len(row) - 1:
                        for j in range(pos + 1, len(row)):
                            new_diffs.add(row[j] - new_val)
                    
                    # Check no conflicts
                    if not new_diffs & temp_used:
                        # Apply
                        row[pos] = new_val
                        cur_used = temp_used | new_diffs
                        improved = True
                        break
        
        new_scope = _scope(cur_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in cur_rows]
            improved = True
        
        if not improved:
            break
    
    return best_rows, best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Initial greedy construction
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng=None, skip_prob=0.0)
        if row is None:
            break
        rows.append(row)
    
    if len(rows) < n_rows:
        # Fallback: simple sequential greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Randomized greedy restarts with local improvement
    rng = random.Random(42)
    
    # Try multiple randomized greedy constructions
    max_restarts = 300
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        # Vary skip probability
        skip_prob = rng.uniform(0.0, 0.3)
        
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng=rng, skip_prob=skip_prob)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if not success or len(rows) < n_rows:
            continue
        
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        
        # Local improvement
        if time.time() < deadline - 5:
            improved_rows, improved_scope = _local_improve(
                rows, k, used, max_iter=20, rng=rng
            )
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = [list(r) for r in improved_rows]
        
        # Early exit if we hit the theoretical lower bound
        # Counting bound: total diffs = n_rows * k*(k+1)/2
        # These must all be distinct and <= scope, so scope >= total_diffs
        total_diffs = n_rows * k * (k + 1) // 2
        if best_scope <= total_diffs:
            break
    
    return best_rows

# EVOLVE-BLOCK-END
