"""Difference triangle set (7,5): Optimized SA with incremental cost and scope minimization."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

def _calculate_diffs(row):
    """Calculate all pairwise differences for a single row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _initial_greedy(n_rows, k):
    """
    Generate an initial valid set using a greedy approach.
    Each row has k+1 elements starting with 0.
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            candidate = row[-1] + 1
            while True:
                new_diffs = {candidate - x for x in row}
                if new_diffs.isdisjoint(used_diffs):
                    row.append(candidate)
                    break
                candidate += 1
        for d in _calculate_diffs(row):
            used_diffs.add(d)
        rows.append(row)
    return rows

def _get_diff_counts(rows):
    """Get a dictionary of difference -> count for all rows."""
    counts = defaultdict(int)
    for row in rows:
        for d in _calculate_diffs(row):
            counts[d] += 1
    return counts

def _count_dups(diff_counts):
    """Count the number of duplicate differences (excess over 1)."""
    return sum(c - 1 for c in diff_counts.values() if c > 1)

def _get_scope(rows):
    return max(max(r) for r in rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 148  # 2s safety margin under 150s limit
    
    # 1. Initialize
    rows = _initial_greedy(n_rows, k)
    diff_counts = _get_diff_counts(rows)
    current_dups = _count_dups(diff_counts)
    current_scope = _get_scope(rows)
    
    best_rows = [list(r) for r in rows]
    best_dups = current_dups
    best_scope = current_scope
    
    rng = random.Random(42)
    
    # Parameters
    temp = 50.0
    cooling = 0.999
    min_temp = 0.5
    
    # Penalty for duplicates is high to ensure validity first
    DUP_PENALTY = 100000
    
    def get_cost(dups, scope):
        return dups * DUP_PENALTY + scope

    current_cost = get_cost(current_dups, current_scope)
    best_cost = current_cost

    while time.time() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        # Select a row and a position to modify
        r_idx = rng.randrange(n_rows)
        row = rows[r_idx]
        m = len(row)
        
        if m <= 1:
            continue
            
        pos = rng.randrange(1, m)
        
        # Determine range for new value
        lo = row[pos - 1] + 1
        hi = row[pos + 1] - 1 if pos + 1 < m else best_scope + 50
        
        if hi < lo:
            continue
            
        # Randomly pick a new value in [lo, hi]
        # Bias towards smaller values to help reduce scope
        # Simple uniform for now
        new_val = rng.randint(lo, hi)
        if new_val == row[pos]:
            continue
            
        old_val = row[pos]
        
        # Calculate change in diff counts
        old_diffs = []
        new_diffs = []
        
        for j in range(m):
            if j == pos:
                continue
            old_d = abs(row[j] - old_val)
            new_d = abs(row[j] - new_val)
            old_diffs.append(old_d)
            new_diffs.append(new_d)
        
        # Calculate change in dups
        delta_dups = 0
        
        # Process removals
        for d in old_diffs:
            c = diff_counts[d]
            if c == 1:
                # 1 -> 0. Dups change: 0 -> 0. Delta 0.
                pass
            else:
                # c -> c-1. Dups change: (c-1) -> (c-2). Delta -1.
                delta_dups -= 1
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Process additions
        for d in new_diffs:
            c = diff_counts.get(d, 0)
            if c == 0:
                # 0 -> 1. Dups change: 0 -> 0. Delta 0.
                pass
            elif c == 1:
                # 1 -> 2. Dups change: 0 -> 1. Delta +1.
                delta_dups += 1
            else:
                # c -> c+1. Dups change: (c-1) -> (c). Delta +1.
                delta_dups += 1
            diff_counts[d] = c + 1
            
        new_dups = current_dups + delta_dups
        
        # Calculate new scope
        # Only recalculate global scope if it might have changed
        # If new_val > current_scope, scope increases.
        # If old_val was the global max and new_val < old_val, scope might decrease.
        
        # Efficient scope update:
        # If new_val > current_scope:
        #   new_scope = new_val
        # Else:
        #   If old_val == current_scope:
        #      # Need to check if any other row has current_scope or if this row has another element == current_scope
        #      # This is expensive. 
        #      # Let's just recalculate scope if dups is 0 or if we are near the best.
        #      # Or, simpler: just use new_val as a lower bound and old_val as upper.
        #      # For cost function, if dups > 0, scope is secondary.
        #      # If dups == 0, we want to minimize scope.
        
        # Let's just calculate the max of the modified row
        new_row_max = max(row[:pos] + [new_val] + row[pos+1:])
        
        # Estimate new global scope
        # If new_row_max > current_scope:
        #    new_scope = new_row_max
        # Else:
        #    # If old_val was the unique global max, scope might drop.
        #    # We'll do a cheap check: if old_val == current_scope, 
        #    # we might need to recalculate. 
        #    # To save time, we'll only recalculate global scope if new_dups == 0 
        #    # and (old_val == current_scope or new_val > current_scope).
        #    # Otherwise, we use current_scope as an approximation (which is safe for 
        #    # accepting moves that don't increase scope, but might miss scope reductions).
        #    # Actually, if we don't update scope correctly, we might think a move is 
        #    # better than it is, or vice versa.
        #    # Let's just recalculate scope every time if new_dups == 0. 
        #    # If new_dups > 0, scope doesn't matter for validity, so we can use 
        #    # current_scope + (new_val - old_val) as a rough proxy? No.
        #    # Let's just use current_scope if new_dups > 0.
        
        if new_dups == 0:
             # Recalculate scope accurately
             # This is O(N*k). With N=7, k=6, it's 42 ops. Very fast.
             new_scope = _get_scope(rows) # Wait, rows is not updated yet.
             # We need to calculate scope with the new value.
             # Let's do it manually.
             max_other = 0
             for i, r in enumerate(rows):
                 if i == r_idx:
                     continue
                 if max(r) > max_other:
                     max_other = max(r)
             new_scope = max(new_row_max, max_other)
        else:
             # If dups > 0, we don't care as much about exact scope for the decision,
             # but we should still track it somewhat.
             # If new_val > current_scope, new_scope = new_val
             # Else, new_scope = current_scope (approximation)
             if new_val > current_scope:
                 new_scope = new_val
             else:
                 new_scope = current_scope
                 
        new_cost = get_cost(new_dups, new_scope)
        old_cost = current_cost
        
        # Acceptance probability
        if new_cost <= old_cost:
            accept = True
        else:
            delta_cost = new_cost - old_cost
            accept = rng.random() < math.exp(-delta_cost / temp)
            
        if accept:
            # Apply move
            row[pos] = new_val
            current_dups = new_dups
            current_scope = new_scope
            current_cost = new_cost
            
            # Update best
            if new_dups < best_dups or (new_dups == best_dups and new_scope < best_scope):
                best_dups = new_dups
                best_scope = new_scope
                best_cost = new_cost
                best_rows = [list(r) for r in rows]
        else:
            # Rollback
            # Restore old diffs
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in old_diffs:
                diff_counts[d] += 1
            row[pos] = old_val
            # current_dups, current_scope, current_cost remain unchanged
            
        temp *= cooling
        
    # Final validation and return
    # Ensure the returned solution is valid
    final_counts = _get_diff_counts(best_rows)
    if _count_dups(final_counts) != 0:
        # Fallback to greedy if SA failed (shouldn't happen if initialized valid and only valid moves accepted? 
        # No, SA accepts invalid moves. So best_rows might be invalid if we only tracked cost.)
        # Wait, I updated best_rows only if dups was lower. 
        # If initial was valid, best_dups starts at 0.
        # If we accept a move that makes dups > 0, best_rows is NOT updated.
        # So best_rows should always be valid (or at least as good as initial).
        # However, if we want to be absolutely safe:
        if _count_dups(_get_diff_counts(best_rows)) != 0:
             return _initial_greedy(n_rows, k)
             
    return best_rows

# EVOLVE-BLOCK-END
