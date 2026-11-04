"""
New strategy: Simulated Annealing with coordinated row-level moves.
Instead of greedy + local search, we start from a feasible solution (possibly loose)
and use SA to minimize the maximum element (scope).
Moves include:
1. Moving a single element within a row (local search style).
2. Swapping elements between two rows (coordinated).
3. Randomly resetting a row to a new random valid configuration (escape local minima).
"""

import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    m = 0
    for d in _row_diffs(row):
        m |= (1 << d)
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _generate_random_row(max_val, num_elements, rng):
    """Generate a random row of num_elements increasing ints starting at 0, max <= max_val."""
    if num_elements == 1:
        return [0]
    if num_elements == 2:
        return [0, rng.randint(1, max_val)]
    
    # We need num_elements - 1 values from 1..max_val
    # Ensure enough space: need at least num_elements-1 distinct values
    if max_val < num_elements - 1:
        return None
        
    # Pick distinct values
    vals = rng.sample(range(1, max_val + 1), num_elements - 1)
    vals.sort()
    return [0] + vals

def _greedy_construct(n_rows, num_elements, max_scope, rng, max_restarts=20):
    """
    Fast greedy to get a valid starting point.
    """
    for restart in range(max_restarts):
        used_mask = 0
        rows = []
        success = True
        for _ in range(n_rows):
            # Try to find a row that fits
            found = False
            for _ in range(100):
                row = _generate_random_row(max_scope, num_elements, rng)
                if row is None:
                    continue
                diffs = _row_diffs(row)
                if len(diffs) != len(set(diffs)):
                    continue
                m = _row_mask(row)
                if m & used_mask:
                    continue
                rows.append(row)
                used_mask |= m
                found = True
                break
            if not found:
                success = False
                break
        if success:
            return rows
    return None

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Counting bound: m(m-1)/2 >= total_diffs_needed -> m ~ sqrt(2*total)
    # Actually, the differences are in 1..M. There are M possible differences.
    # We need total_diffs_needed distinct differences.
    # So M >= total_diffs_needed.
    # But the largest entry is M.
    # Wait, the differences are a_j - a_i. Max diff is M.
    # So we need M >= total_diffs_needed.
    # Counting bound: M >= n_rows * k(k+1)/2.
    # For (7,5): 7 * 15 = 105.
    counting_bound = n_rows * num_pairs
    
    # Best known is 112.
    # Let's start SA with a scope slightly above counting bound, or best known if known.
    # Since we don't know n_rows, k generally, we use counting bound + margin.
    initial_scope = counting_bound + 20
    
    rng = random.Random(42)
    
    # Phase 1: Get a valid initial solution
    best_rows = None
    best_scope = float('inf')
    
    # Try to find a valid solution with scope around counting_bound
    # If we can't, expand scope.
    scope_try = initial_scope
    while scope_try < 300 and time.time() < DEADLINE - 30:
        rows = _greedy_construct(n_rows, num_elements, scope_try, rng, max_restarts=10)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            # If we found one, we can try to find a better one with smaller scope?
            # No, greedy with smaller scope is harder.
            # We found one with scope <= scope_try.
            # Let's break and refine.
            break
        scope_try += 5
        
    if best_rows is None:
        # Fallback: very loose greedy
        scope_try = 500
        rows = _greedy_construct(n_rows, num_elements, scope_try, rng, max_restarts=50)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)
        else:
            # Sequential fallback
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            best_rows = rows
            best_scope = _scope(rows)

    # Phase 2: Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Precompute masks for speed
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
        
    def get_global_mask():
        gm = 0
        for m in row_masks:
            gm |= m
        return gm

    T = 10.0
    T_min = 0.01
    alpha = 0.999
    max_iter = 500000
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        # Pick a random move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Type 1: Move a single element within a row
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, len(current_rows[row_idx]) - 1)
            old_val = current_rows[row_idx][pos]
            
            # Determine range for new_val
            low = current_rows[row_idx][pos - 1] + 1
            high = current_rows[row_idx][pos + 1] - 1 if pos < len(current_rows[row_idx]) - 1 else current_scope + 5
            
            if low >= high:
                continue
                
            # Bias towards reducing scope
            if old_val == current_scope and rng.random() < 0.5:
                # Try to move max element down
                high = min(high, old_val - 1)
                if low >= high:
                    continue
            
            new_val = rng.randint(low, high)
            if new_val == old_val:
                continue
            
            # Try move
            row = current_rows[row_idx][:]
            row[pos] = new_val
            diffs = _row_diffs(row)
            if len(diffs) != len(set(diffs)):
                continue
            new_mask = _row_mask(row)
            other_mask = global_mask & ~row_masks[row_idx]
            if new_mask & other_mask:
                continue
            
            # Calculate delta scope
            old_scope = current_scope
            new_scope = max(old_scope, new_val)
            # If we moved the max element, scope might decrease
            if new_val < old_scope:
                # Check if any other row has this max
                # Actually, scope is max of all rows.
                # If we reduce an element that was the unique max, scope drops.
                # If not, scope stays.
                # Simplified: new_scope = max(new_val, max of other rows' last elements)
                # But we don't track max of others easily.
                # Let's just compute new_scope properly for acceptance?
                # For speed, approximate:
                # If old_val was the max, and new_val < old_val,
                # new_scope is either old_scope (if another row has it) or new_val (if not).
                # Let's just compute it if it looks promising.
                # Actually, computing scope is O(n_rows).
                # Let's do it.
                max_other = 0
                for i in range(n_rows):
                    if i != row_idx:
                        if current_rows[i][-1] > max_other:
                            max_other = current_rows[i][-1]
                new_scope = max(new_val, max_other)
            else:
                new_scope = old_scope
                
            delta = new_scope - old_scope
            
            # Acceptance
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Apply
                current_rows[row_idx][pos] = new_val
                row_masks[row_idx] = new_mask
                global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                    
        elif move_type < 0.8:
            # Type 2: Swap elements between two rows
            if n_rows < 2:
                continue
            r1, r2 = rng.sample(range(n_rows), 2)
            p1 = rng.randint(1, len(current_rows[r1]) - 1)
            p2 = rng.randint(1, len(current_rows[r2]) - 1)
            
            v1 = current_rows[r1][p1]
            v2 = current_rows[r2][p2]
            
            # Check if swap maintains increasing order
            # Row 1: ... < v2 < ...
            if p1 > 1 and v2 <= current_rows[r1][p1-1]: continue
            if p1 < len(current_rows[r1])-1 and v2 >= current_rows[r1][p1+1]: continue
            # Row 2: ... < v1 < ...
            if p2 > 1 and v1 <= current_rows[r2][p2-1]: continue
            if p2 < len(current_rows[r2])-1 and v1 >= current_rows[r2][p2+1]: continue
            
            # Try swap
            row1 = current_rows[r1][:]
            row2 = current_rows[r2][:]
            row1[p1] = v2
            row2[p2] = v1
            
            diffs1 = _row_diffs(row1)
            if len(diffs1) != len(set(diffs1)): continue
            diffs2 = _row_diffs(row2)
            if len(diffs2) != len(set(diffs2)): continue
            
            m1 = _row_mask(row1)
            m2 = _row_mask(row2)
            
            other_mask = global_mask & ~row_masks[r1] & ~row_masks[r2]
            if (m1 | m2) & other_mask: continue
            
            # Check internal conflicts between m1 and m2?
            # No, they are in different rows, so their diffs are distinct by row separation.
            # Wait, the condition is "all positive differences ... are distinct over all rows".
            # This means diff_a from row 1 cannot equal diff_b from row 2.
            # So m1 and m2 must be disjoint.
            if m1 & m2: continue
            
            # Scope change
            old_scope = current_scope
            new_scope = max(row1[-1], row2[-1], current_scope) # Approx
            # Actually, if we swap, the last elements might change if p1 or p2 is last.
            # If p1 is last, row1[-1] changes to v2.
            # Let's compute new_scope properly.
            max_val = 0
            for i in range(n_rows):
                if i == r1:
                    if row1[-1] > max_val: max_val = row1[-1]
                elif i == r2:
                    if row2[-1] > max_val: max_val = row2[-1]
                else:
                    if current_rows[i][-1] > max_val: max_val = current_rows[i][-1]
            new_scope = max_val
            
            delta = new_scope - old_scope
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Apply
                current_rows[r1][p1] = v2
                current_rows[r2][p2] = v1
                row_masks[r1] = m1
                row_masks[r2] = m2
                global_mask = (global_mask & ~m1 & ~m2) | m1 | m2
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                    
        else:
            # Type 3: Reset a row
            row_idx = rng.randint(0, n_rows - 1)
            # Generate a new row that is compatible with others
            # To ensure compatibility, we should avoid used diffs.
            # But we don't track used diffs as a set, only mask.
            # We can try to generate a row that doesn't conflict.
            
            # Try a few times
            for _ in range(10):
                # Limit scope to current_scope or slightly less
                max_val = current_scope
                if rng.random() < 0.5:
                    max_val = max(1, current_scope - 5)
                    
                new_row = _generate_random_row(max_val, num_elements, rng)
                if new_row is None:
                    continue
                diffs = _row_diffs(new_row)
                if len(diffs) != len(set(diffs)):
                    continue
                new_mask = _row_mask(new_row)
                other_mask = global_mask & ~row_masks[row_idx]
                if new_mask & other_mask:
                    continue
                
                # Accept if scope doesn't increase too much or randomly
                new_scope = max(new_row[-1], current_scope)
                # Actually, if we replace a row, the new scope is max(new_row[-1], max of others)
                max_other = 0
                for i in range(n_rows):
                    if i != row_idx:
                        if current_rows[i][-1] > max_other:
                            max_other = current_rows[i][-1]
                new_scope = max(new_row[-1], max_other)
                
                delta = new_scope - current_scope
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_rows[row_idx] = new_row
                    row_masks[row_idx] = new_mask
                    global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                    break
            
        T *= alpha
        if T < T_min:
            T = T_min

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
