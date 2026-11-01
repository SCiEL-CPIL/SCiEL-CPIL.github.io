import time
import sys
import random
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach:
    1. Fast greedy initialization.
    2. Row-level re-optimization to tighten individual rows.
    3. Simulated Annealing with row-swap and element-nudge moves.
    4. Periodic full-row re-optimization for the bottleneck row.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    # Total number of differences needed
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Helper: Calculate all differences for a single row
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: Calculate all differences for a set of rows
    def _all_diffs(rows):
        all_d = set()
        for r in rows:
            all_d |= _row_diffs(r)
        return all_d

    # Helper: Global max
    def _global_max(rows):
        return max(max(r) for r in rows)

    # 1. Greedy Construction (Fast baseline)
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Search for next element
                # Start from current_max + 1, but maybe skip ahead if tight
                # Simple linear scan
                x = current_max + 1
                found = False
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        current_max = x
                        found = True
                        break
                    x += 1
                if not found:
                    return None
            rows.append(row)
        return rows

    # Get initial solution
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_scope = _global_max(best_solution)
    best_rows = [r[:] for r in best_solution]

    # 2. Row Re-optimization
    # Given a row and a set of forbidden differences (from other rows),
    # try to minimize the max element of the row while maintaining internal validity
    # and avoiding forbidden diffs.
    def reoptimize_row(row, forbidden_diffs, k, max_scope_limit=None):
        n = k + 1
        best_row = list(row)
        best_max = max(best_row)
        
        if max_scope_limit is not None and best_max > max_scope_limit:
            # If we are already worse than the limit, we might still try to improve
            # but we won't accept if we can't beat the limit? 
            # Actually, we want to find a row that fits within max_scope_limit if possible.
            # If current row exceeds it, we must find a better one.
            pass

        forbidden = set(forbidden_diffs)
        
        # Iterative local search on the row
        # We try to reduce each element from right to left
        for iteration in range(50):
            improved = False
            # Iterate from last element to first (index 1 to n-1)
            for pi in range(n - 1, 0, -1):
                old_val = best_row[pi]
                min_allowed = best_row[pi - 1] + 1
                # Max allowed is constrained by next element if exists, or max_scope_limit
                max_allowed = best_row[pi + 1] - 1 if pi < n - 1 else (max_scope_limit - 1 if max_scope_limit else old_val)
                
                if old_val <= min_allowed:
                    continue

                # Try to decrease this element
                # We scan from old_val - 1 down to min_allowed
                # To speed up, we can check validity quickly
                for new_val in range(old_val - 1, min_allowed - 1, -1):
                    if new_val < min_allowed:
                        break
                    if pi < n - 1 and new_val >= best_row[pi + 1]:
                        continue # Wait, if we are decreasing, new_val < old_val. 
                                 # If old_val was valid, new_val might violate new_val < best_row[pi+1]?
                                 # No, if we are decreasing best_row[pi], it moves further from best_row[pi+1].
                                 # So new_val < best_row[pi+1] is always true if old_val was.
                                 # Actually, we need new_val > best_row[pi-1]. That's min_allowed.
                                 # And we need new_val < best_row[pi+1] if pi < n-1.
                                 # Since we are decreasing, new_val < old_val < best_row[pi+1] (if valid before).
                                 # So the only constraint is new_val > best_row[pi-1].
                                 # Wait, internal diffs must be unique.
                                 # Let's check validity.
                    
                    cand = best_row[:]
                    cand[pi] = new_val
                    
                    # Check internal validity and forbidden diffs
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = cand[j] - cand[i]
                            if d in new_diffs:
                                valid = False
                                break
                            if d in forbidden:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid:
                            break
                    
                    if valid:
                        best_row = cand
                        best_max = max(best_row)
                        improved = True
                        break # Restart scan for this position or next?
                        # If we moved, the constraints for previous positions (i < pi) change?
                        # Diffs involving pi change. Diffs not involving pi stay same.
                        # We should probably restart the loop for this pi or continue to pi-1.
                        # Since we are going right to left, and we changed pi, 
                        # diffs for pi-1 involving pi changed. So we should continue to pi-1.
                        # The 'break' exits the new_val loop, then the for pi loop continues to pi-1.
                        # This is correct.
                
                if improved:
                    continue
            if not improved:
                break
        
        return best_row

    # 3. Simulated Annealing
    # State: list of rows
    # Cost: global max
    # Moves:
    # a) Nudge an element in a row (try to reduce it)
    # b) Re-optimize a specific row entirely
    
    current_rows = [r[:] for r in best_solution]
    current_val = _global_max(current_rows)
    
    # Pre-compute diffs for efficiency? 
    # For n_rows=7, k=5, total diffs ~ 105. Sets are small.
    # We can maintain a global set of used diffs and row-specific sets.
    
    row_diffs = [_row_diffs(r) for r in current_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
        
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.9999
    
    no_improve_count = 0
    max_no_improve = 5000
    
    last_reopt_time = time.time()
    reopt_interval = 0.5 # seconds
    
    while time.time() < deadline - 5.0:
        if no_improve_count > max_no_improve:
            # Restart with a different random seed or slightly perturbed solution
            # Perturb: pick a random row, random element, change it slightly?
            # Or just reset T and restart SA
            T = 1000.0
            no_improve_count = 0
            # Maybe shuffle rows? They are indistinguishable, but order in list doesn't matter for validity.
            # However, for re-optimization bias, it might matter.
            # Let's just continue.
            
        # Periodic Row Re-optimization
        now = time.time()
        if now - last_reopt_time > reopt_interval:
            last_reopt_time = now
            # Identify the row with the maximum value
            max_row_idx = max(range(n_rows), key=lambda i: max(current_rows[i]))
            # Get forbidden diffs (all diffs except those in this row)
            forbidden = all_diffs - row_diffs[max_row_idx]
            
            # Try to re-optimize this row to minimize its max, constrained by forbidden diffs
            # We aim to keep the global max low.
            # The target max for this row should be <= current global max.
            # Actually, we want to see if we can reduce the global max.
            # So we try to make this row's max < current global max if it's the bottleneck.
            # Or just minimize it.
            
            # Limit the search scope of re-optimization to current global max
            # If the row's max is already > current global max, it's the bottleneck.
            # We want to find a valid row with max < current global max (if possible)
            # or at least <= current global max.
            
            new_row = reoptimize_row(current_rows[max_row_idx], forbidden, k, max_scope_limit=current_val)
            
            # Check if new_row is valid (it should be by construction) and if it improves global max
            new_diffs = _row_diffs(new_row)
            # Ensure no conflict with other rows
            if not (new_diffs & (all_diffs - row_diffs[max_row_idx])):
                new_max = max(new_row)
                new_global_max = max([max(r) for i, r in enumerate(current_rows) if i != max_row_idx] + [new_max])
                
                if new_global_max < current_val:
                    # Accept
                    all_diffs -= row_diffs[max_row_idx]
                    all_diffs |= new_diffs
                    current_rows[max_row_idx] = new_row
                    row_diffs[max_row_idx] = new_diffs
                    current_val = new_global_max
                    no_improve_count = 0
                    if current_val < best_scope:
                        best_scope = current_val
                        best_rows = [r[:] for r in current_rows]
                elif new_global_max == current_val and random.random() < 0.1:
                    # Neutral move, small probability
                    all_diffs -= row_diffs[max_row_idx]
                    all_diffs |= new_diffs
                    current_rows[max_row_idx] = new_row
                    row_diffs[max_row_idx] = new_diffs
                    # current_val stays same
                else:
                    # Reject
                    pass
        
        # SA Move: Nudge a random element
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        # Pick a random position (1 to k)
        p_idx = random.randint(1, k)
        old_val = row[p_idx]
        
        # Try to change it to a nearby value
        # Bias towards decreasing
        if random.random() < 0.7:
            # Decrease
            min_allowed = row[p_idx - 1] + 1
            max_allowed = old_val - 1
            if min_allowed <= max_allowed:
                new_val = random.randint(min_allowed, max_allowed)
            else:
                continue
        else:
            # Increase (to escape local minima)
            min_allowed = old_val + 1
            # Don't increase too much, keep it reasonable
            max_allowed = old_val + 10
            if p_idx < k:
                max_allowed = min(max_allowed, row[p_idx + 1] - 1)
            if min_allowed <= max_allowed:
                new_val = random.randint(min_allowed, max_allowed)
            else:
                continue
        
        # Check validity
        cand = row[:]
        cand[p_idx] = new_val
        
        # Check internal validity and global conflicts
        valid = True
        new_row_diffs = set()
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = cand[j] - cand[i]
                if d in new_row_diffs:
                    valid = False
                    break
                new_row_diffs.add(d)
            if not valid:
                break
        
        if valid:
            # Check against other rows
            # Diffs from other rows: all_diffs - row_diffs[r_idx]
            other_diffs = all_diffs - row_diffs[r_idx]
            if not (new_row_diffs & other_diffs):
                # Valid move
                new_max = max(cand)
                # Calculate new global max
                # We only changed row r_idx.
                # Other rows' maxes are unchanged.
                # We can maintain a list of row maxes for speed?
                # For now, just calculate.
                other_maxes = [max(current_rows[i]) for i in range(n_rows) if i != r_idx]
                new_global_max = max(other_maxes + [new_max])
                
                delta = new_global_max - current_val
                if delta <= 0:
                    # Improvement or neutral
                    # Accept
                    all_diffs -= row_diffs[r_idx]
                    all_diffs |= new_row_diffs
                    current_rows[r_idx] = cand
                    row_diffs[r_idx] = new_row_diffs
                    current_val = new_global_max
                    no_improve_count = 0
                    if current_val < best_scope:
                        best_scope = current_val
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Worse
                    if random.random() < math.exp(-delta / T):
                        all_diffs -= row_diffs[r_idx]
                        all_diffs |= new_row_diffs
                        current_rows[r_idx] = cand
                        row_diffs[r_idx] = new_row_diffs
                        current_val = new_global_max
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_rows

