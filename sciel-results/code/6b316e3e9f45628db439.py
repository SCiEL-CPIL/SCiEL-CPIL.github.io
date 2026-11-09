"""
Difference Triangle Set (7,5) Solver
Combines interleaved greedy initialization with simulated annealing row replacement.
"""

import time
import random
import math

# Global deadline guard
_START_TIME = time.time()
# Evaluator kills at 150s. We stop searching at 140s to allow for cleanup and return.
_DEADLINE = _START_TIME + 140.0

def _dead():
    return time.time() >= _DEADLINE

def _get_diffs(row):
    """Calculate the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs

def _scope(rows):
    """Calculate the scope (maximum entry) of a set of rows."""
    if not rows:
        return float('inf')
    return max(max(r) for r in rows)

def _build_interleaved(n_rows, k, max_scope, rnd, randomize_prob=0.0):
    """
    Greedily build rows interleaved: extend the row with the smallest current maximum.
    This keeps rows balanced and compact.
    max_scope: Hard limit for the maximum entry.
    randomize_prob: Probability of skipping a valid candidate to explore other options.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    # We need to add k elements to each row (total n_rows * k additions)
    # We track which rows are complete
    complete = [False] * n_rows
    
    for _ in range(n_rows * k):
        # Find incomplete rows
        candidates = [i for i in range(n_rows) if not complete[i]]
        if not candidates:
            break
            
        # Sort candidates by their current last element (smallest first)
        # This is the "interleaved" strategy: grow the shortest row first
        candidates.sort(key=lambda idx: rows[idx][-1])
        
        # Select a row to extend
        # With probability randomize_prob, pick a random candidate from the top few
        # to allow for some diversity in the greedy choice
        if randomize_prob > 0 and len(candidates) > 1 and rnd.random() < randomize_prob:
            # Pick from top 3 smallest
            top_k = min(3, len(candidates))
            i = rnd.choice(candidates[:top_k])
        else:
            i = candidates[0]
        
        row = rows[i]
        last_val = row[-1]
        found = False
        
        # Try to find the next value x > last_val such that all new diffs are unused
        # and x <= max_scope
        for x in range(last_val + 1, max_scope + 1):
            # Calculate new differences introduced by x
            new_diffs = set()
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Check internal consistency (though x > last_val ensures positive diffs)
                # and that we don't exceed scope
                if x <= max_scope:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
        
        if not found:
            return None
        
        if len(row) == k + 1:
            complete[i] = True

    # Verify all rows are complete
    if any(not c for c in complete):
        return None
        
    return rows

def _construct_row_greedy(used_diffs, k, max_mark, rnd, randomize_prob=0.0):
    """
    Construct a single row of length k+1 starting at 0, respecting used_diffs and max_mark.
    Returns (row, diffs_set) or (None, None) if impossible.
    """
    row = [0]
    local_diffs = set()
    
    while len(row) < k + 1:
        last_val = row[-1]
        found = False
        
        # We can randomize the search for the next element to escape local minima
        # Start from last_val + 1
        # If randomize_prob is high, we might skip some valid small numbers
        
        candidates = list(range(last_val + 1, max_mark + 1))
        
        # To allow randomization, we might shuffle the first few candidates or skip some
        # Simple strategy: iterate through, but with probability p, skip a valid candidate
        # to see if a larger one allows better packing later.
        # However, for speed, we stick to sequential search but allow "jumps" if stuck?
        # Actually, standard greedy is fastest. Let's add a slight randomization:
        # With probability randomize_prob, pick a random x in [last_val+1, max_mark]
        # and check if it works. If not, fall back to sequential.
        
        if randomize_prob > 0 and rnd.random() < randomize_prob:
            # Try a random candidate
            x = rnd.randint(last_val + 1, max_mark)
            new_diffs = set()
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                row.append(x)
                local_diffs.update(new_diffs)
                found = True
        
        if not found:
            # Sequential greedy
            for x in range(last_val + 1, max_mark + 1):
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    row.append(x)
                    local_diffs.update(new_diffs)
                    found = True
                    break
        
        if not found:
            return None, None
            
    return row, local_diffs

def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')
    
    # Lower bound on scope from counting argument:
    # Total differences = n_rows * k * (k+1) / 2
    # Max difference is m. Number of available differences is m * (m+1) / 2? No, differences are in [1, m].
    # Actually, the set of all differences must be a subset of {1, ..., m}.
    # So m * (m+1) / 2 >= n_rows * k * (k+1) / 2
    # m(m+1) >= n_rows * k * (k+1)
    lb = 1
    while lb * (lb + 1) < n_rows * k * (k + 1):
        lb += 1
        
    # --- Phase 1: Initialization with Interleaved Greedy ---
    
    # Try several random seeds for interleaved greedy to find a good starting point
    # The scope limit for the initial build can be tight (lb) or loose (lb + margin)
    
    initial_scopes = [lb, lb + 5, lb + 10, lb + 20, lb + 50]
    
    for scope_limit in initial_scopes:
        if _dead():
            break
        for seed in range(5):
            if _dead():
                break
            local_rnd = random.Random(seed * 100 + scope_limit)
            rows = _build_interleaved(n_rows, k, scope_limit, local_rnd, randomize_prob=0.2)
            if rows is not None:
                sc = _scope(rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [r[:] for r in rows]
                # If we found a solution at the lower bound, we can stop early
                if sc == lb:
                    break
        if best_scope == lb:
            break

    if best_rows is None:
        # Fallback: Sequential greedy with a very large scope
        # This should always succeed but might have a large scope
        used = set()
        rows = []
        for _ in range(n_rows):
            row, diffs = _construct_row_greedy(used, k, 1000, rnd)
            if row is None:
                # Last resort: trivial construction (might be invalid but structural)
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    row.append(x)
                diffs = _get_diffs(row)
            rows.append(row)
            used.update(diffs)
        best_rows = rows
        best_scope = _scope(best_rows)
        
    if _dead():
        return best_rows

    # --- Phase 2: Simulated Annealing ---
    
    # Copy current best
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Pre-compute diffs for current rows
    row_diffs = [_get_diffs(r) for r in current_rows]
    all_diffs = set()
    for d in row_diffs:
        all_diffs.update(d)
        
    best_rows_sa = [r[:] for r in current_rows]
    best_scope_sa = current_scope
    
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    # Track no-improve steps to reset or stop
    no_improve = 0
    max_no_improve = 5000
    
    while not _dead() and T > T_min:
        T *= cooling
        
        # Pick a row to replace
        # Bias towards rows that have the maximum value, as reducing them is most impactful
        max_val = current_scope
        # Find indices of rows with max value
        max_indices = [i for i, r in enumerate(current_rows) if r[-1] == max_val]
        
        if max_indices and rnd.random() < 0.7:
            ri = rnd.choice(max_indices)
        else:
            ri = rnd.randint(0, n_rows - 1)
            
        old_row = current_rows[ri]
        old_diffs = row_diffs[ri]
        
        # Remove old row's diffs from global set
        # Note: We must be careful. If a diff appears in multiple rows, we can't just discard.
        # But the problem states "all differences... are distinct over all rows".
        # This implies the sets of differences for each row are disjoint!
        # So we can safely remove.
        for d in old_diffs:
            all_diffs.discard(d)
            
        # Determine target max for the new row
        # We want to reduce the scope if possible.
        # If this row was the max, we MUST reduce it to improve global scope.
        # If not, we can try to keep it same or reduce.
        
        if old_row[-1] == current_scope:
            # Try to find a row with max < current_scope
            target_max = current_scope - 1
        else:
            # Try to find a row with max <= old_row[-1]
            target_max = old_row[-1]
            
        # Try to construct a new row
        new_row, new_diffs = _construct_row_greedy(all_diffs, k, target_max, rnd, randomize_prob=0.1)
        
        if new_row is None:
            # Try with a slightly larger target if we were trying to reduce
            # or just revert
            if old_row[-1] == current_scope:
                # If we couldn't reduce, try to keep same scope? 
                # No, if we can't reduce, we might as well keep the old row.
                # But maybe a different row with same scope helps future reductions?
                # For now, revert.
                pass
            else:
                # Try to construct with same scope
                new_row, new_diffs = _construct_row_greedy(all_diffs, k, target_max, rnd, randomize_prob=0.3)
                
        if new_row is not None:
            # Calculate new global scope
            # The new scope is max of (max of other rows, max of new row)
            other_maxes = [current_rows[i][-1] for i in range(n_rows) if i != ri]
            new_global_scope = max(max(other_maxes), new_row[-1])
            
            # Acceptance criteria
            if new_global_scope < current_scope:
                # Always accept improvement
                current_rows[ri] = new_row
                row_diffs[ri] = new_diffs
                all_diffs.update(new_diffs)
                current_scope = new_global_scope
                
                if current_scope < best_scope_sa:
                    best_scope_sa = current_scope
                    best_rows_sa = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # new_global_scope >= current_scope
                # Accept with probability exp(-delta / T)
                delta = new_global_scope - current_scope
                if delta <= 0:
                    accept_prob = 1.0
                else:
                    accept_prob = math.exp(-delta / max(T, 0.1))
                
                if rnd.random() < accept_prob:
                    # Accept move even if scope doesn't improve (or worsens slightly)
                    # This helps escape local minima
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    all_diffs.update(new_diffs)
                    current_scope = new_global_scope
                    no_improve += 1
                else:
                    # Reject, revert
                    pass
        else:
            # Could not construct any new row, revert
            pass
            
        # If we rejected, we need to ensure state is consistent.
        # In the "Reject" case above, we didn't modify current_rows, row_diffs, or all_diffs.
        # In the "Accept" case, we did.
        # So state is consistent.
        
        if no_improve > max_no_improve:
            # Restart with best solution found so far
            current_rows = [r[:] for r in best_rows_sa]
            current_scope = best_scope_sa
            row_diffs = [_get_diffs(r) for r in current_rows]
            all_diffs = set()
            for d in row_diffs:
                all_diffs.update(d)
            no_improve = 0
            T = max(T, 5.0) # Reheat slightly

    # Final check: update best if SA found something better
    if best_scope_sa < best_scope:
        best_scope = best_scope_sa
        best_rows = [r[:] for r in best_rows_sa]
        
    return best_rows
