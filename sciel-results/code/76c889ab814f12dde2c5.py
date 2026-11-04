"""Difference triangle set (4,7) – optimized construction.

Strategy:
  1. Fast greedy with bitset conflict tracking to get a quick feasible solution.
  2. High-performance Simulated Annealing with incremental difference counting
     to allow O(k) move evaluations instead of O(k^2).
  3. Deadline guard to return the best valid solution before the 120 s limit.
"""

import time
import random
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_diffs(row):
    """Return the set of positive differences for a row (list of ints)."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs

def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for r in rows:
        d = _row_diffs(r)
        if d & seen:
            return False
        seen |= d
    return True

# ---------------------------------------------------------------------------
# Greedy construction (fast baseline)
# ---------------------------------------------------------------------------

def _greedy(n_rows, k):
    """Greedy: build rows one at a time, smallest next mark with no conflict."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = set()
            conflict = False
            for a in row:
                d = x - a
                if d in used:
                    conflict = True
                    break
                new.add(d)
            if not conflict:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# Fast Simulated Annealing
# ---------------------------------------------------------------------------

def _fast_sa(rows, n_rows, k, time_limit, rng):
    """
    Simulated Annealing with incremental diff counting.
    """
    # Deep copy
    cur_rows = [list(r) for r in rows]
    cur_scope = _scope(cur_rows)
    
    # Precompute initial diff counts
    # diff_counts[d] = count of difference d across all rows
    max_diff = cur_scope
    # We need an array or dict large enough. Max diff is scope.
    # To be safe, use a dict or a large array. Scope is ~134. Array is fast.
    # Let's use a dict for flexibility, or array of size scope+1.
    # Since scope changes, we might need to resize or just use a dict.
    # Dict is fine for k=7 (28 diffs per row, 112 total).
    
    def build_diff_counts(rs):
        counts = {}
        for r in rs:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i+1, n):
                    d = r[j] - ri
                    counts[d] = counts.get(d, 0) + 1
        return counts

    cur_diff_counts = build_diff_counts(cur_rows)
    
    # Check validity (should be valid if input is valid)
    # A state is valid if all counts are <= 1.
    
    def is_valid_state(counts):
        for v in counts.values():
            if v > 1:
                return False
        return True

    # Initial validity check
    if not is_valid_state(cur_diff_counts):
        # Fallback to greedy if starting point is invalid (shouldn't happen)
        pass

    best_rows = [list(r) for r in cur_rows]
    best_scope = cur_scope
    best_diff_counts = cur_diff_counts.copy()

    T = 50.0 # Initial temperature
    T_min = 0.1
    cooling = 0.99999 # Slow cooling to allow exploration
    
    start_time = time.time()
    deadline = start_time + time_limit

    # Number of moves per second to tune? Let's just run until deadline.
    
    while time.time() < deadline:
        # Pick a random move
        # Move: change rows[ri][ci] to a new value
        ri = rng.randrange(n_rows)
        ci = rng.randrange(1, len(cur_rows[ri])) # Don't change 0 (index 0)
        
        old_val = cur_rows[ri][ci]
        
        # Determine range for new value
        # Must be > cur_rows[ri][ci-1] and < cur_rows[ri][ci+1] (if ci < len-1)
        # Or if ci is last, can be anything > prev.
        # To reduce scope, bias towards smaller values?
        # Let's sample from a range around old_val or from [min_val, max_val]
        
        min_val = cur_rows[ri][ci-1] + 1
        if ci < len(cur_rows[ri]) - 1:
            max_val = cur_rows[ri][ci+1] - 1
        else:
            # Allow expanding? Or just shrink?
            # If we want to minimize scope, we should try to keep max small.
            # But we need to be able to escape local minima.
            # Let's allow values up to current scope + 10
            max_val = max(cur_scope + 10, old_val + 10)
            
        if min_val > max_val:
            continue
            
        # Sample new value
        # Uniform or Gaussian? Uniform is simpler.
        new_val = rng.randint(min_val, max_val)
        if new_val == old_val:
            continue

        # Calculate delta in diff counts
        # Remove old diffs involving index ci
        row = cur_rows[ri]
        removed_diffs = []
        for i in range(len(row)):
            if i == ci:
                continue
            d = abs(row[ci] - row[i])
            removed_diffs.append(d)
            
        # Add new diffs involving index ci
        added_diffs = []
        for i in range(len(row)):
            if i == ci:
                continue
            d = abs(new_val - row[i])
            added_diffs.append(d)

        # Check if move is valid
        # A move is valid if no diff count exceeds 1 after the move.
        # We can check this by looking at the counts of removed and added diffs.
        
        # For each diff in added_diffs, if its count (after removing old ones) would become > 1, it's invalid.
        # Note: if a diff is in both removed and added, the net change is 0.
        
        # Let's simulate the count changes on a temporary basis or check carefully.
        # counts[d] is current count.
        # For a diff d in added_diffs:
        #   new_count = counts[d] - (1 if d in removed_diffs else 0) + 1
        #   if new_count > 1: invalid
        
        valid = True
        # Optimization: if new_val creates a diff that already has count >= 1 and is not removed, it's invalid.
        # More precisely:
        # For each d in added_diffs:
        #   current_count = cur_diff_counts.get(d, 0)
        #   if d in removed_diffs:
        #       # The count will decrease by 1 (removal) and increase by 1 (addition) -> net 0 change for this specific d?
        #       # Wait, removed_diffs might contain duplicates? No, row elements are unique.
        #       # But a diff value might appear multiple times in removed_diffs? No, row is strictly increasing, so diffs are unique within a row?
        #       # Yes, within a single row, all differences are distinct.
        #       # So d appears at most once in removed_diffs.
        #       # So if d is in removed_diffs, the net change for d is 0.
        #       # So we don't need to check d if it's in removed_diffs?
        #       # Wait, what if d appears in added_diffs AND removed_diffs?
        #       # Then the count stays the same.
        #       # What if d appears in added_diffs but NOT in removed_diffs?
        #       # Then count increases by 1.
        #       # What if d appears in removed_diffs but NOT in added_diffs?
        #       # Then count decreases by 1.
        
        # So, the move is invalid if there exists a diff d such that:
        # 1. d is in added_diffs
        # 2. d is NOT in removed_diffs
        # 3. cur_diff_counts[d] >= 1 (because it will become 2)
        
        # Also, we must ensure the row remains strictly increasing.
        # We already enforced min_val/max_val, but let's double check.
        if new_val <= row[ci-1]:
            continue
        if ci < len(row) - 1 and new_val >= row[ci+1]:
            continue
            
        # Build a set of removed_diffs for O(1) lookup
        removed_set = set(removed_diffs)
        
        for d in added_diffs:
            if d not in removed_set:
                if cur_diff_counts.get(d, 0) >= 1:
                    valid = False
                    break
        
        if not valid:
            # Reject invalid move (or accept with probability if we allow invalid states? 
            # Standard SA for constraints usually rejects invalid states or penalizes them.
            # Given the strict constraint, rejecting is safer and simpler.
            # However, this might get stuck if the current state is valid but isolated.
            # To escape, we might need to allow temporary invalid states.
            # Let's stick to valid moves only for simplicity and robustness, 
            # but this might be slow if the space is fragmented.
            # Given the "Difference Triangle" structure, the space is likely connected enough for valid moves.
            continue

        # Move is valid. Calculate delta in scope.
        old_scope = _scope(cur_rows) # This is expensive. 
        # We can track the max value in each row.
        # Let's just compute it for now, k is small.
        
        # Apply move
        cur_rows[ri][ci] = new_val
        
        # Update diff counts
        for d in removed_diffs:
            cur_diff_counts[d] -= 1
            if cur_diff_counts[d] == 0:
                del cur_diff_counts[d]
        for d in added_diffs:
            cur_diff_counts[d] = cur_diff_counts.get(d, 0) + 1
        
        new_scope = _scope(cur_rows)
        
        # Acceptance criterion
        # We want to minimize scope.
        # If new_scope < old_scope, accept.
        # If new_scope > old_scope, accept with probability exp(-delta/T).
        # If new_scope == old_scope, accept with probability 0.5 (to explore).
        
        delta = new_scope - old_scope
        
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rng.random() < 0.5
        else:
            accept = rng.random() < pow(2.718281828, -delta / T)
            
        if accept:
            # Update best
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in cur_rows]
                best_diff_counts = cur_diff_counts.copy()
        else:
            # Revert move
            cur_rows[ri][ci] = old_val
            # Revert diff counts
            for d in added_diffs:
                cur_diff_counts[d] -= 1
                if cur_diff_counts[d] == 0:
                    del cur_diff_counts[d]
            for d in removed_diffs:
                cur_diff_counts[d] = cur_diff_counts.get(d, 0) + 1

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Optionally restart with higher T if stuck? 
            # For now, just keep going.

    return best_rows, best_scope

# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences across all rows are distinct. Minimize the scope.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 s to leave 5 s margin before 120 s kill

    rng = random.Random(42)

    # Phase 1: Quick greedy to get a valid baseline
    rows = _greedy(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)

    # Phase 2: Fast Simulated Annealing
    if time.time() < deadline:
        sa_rows, sa_scope = _fast_sa(best_rows, n_rows, k, deadline - start_time, rng)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # Final validation
    if not _is_valid(best_rows):
        # Fallback to simple greedy
        best_rows = _greedy(n_rows, k)

    return best_rows
