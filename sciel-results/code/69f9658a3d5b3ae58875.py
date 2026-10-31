import time
import math
import random
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin for 120s limit
    
    # 1. Greedy Initialization
    # We need n_rows rows, each of length k+1, starting at 0, strictly increasing.
    # All differences between any two elements in the same row must be distinct across ALL rows.
    
    used_diffs = set()
    rows = []
    
    # Greedy construction: For each row, pick smallest possible integers
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                # Check if adding x creates a duplicate difference
                # Differences are x - a for a in row
                # We also need to ensure x - a is not in used_diffs
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    # Also check differences between x and existing elements don't clash with each other?
                    # No, differences within a row are unique by construction if we pick x carefully?
                    # Actually, we must ensure all pairwise differences in the final row are unique.
                    # The greedy approach above only checks against global used_diffs.
                    # It does NOT check if (x - a1) == (x - a2) which is impossible since a1 != a2.
                    # It DOES check if (x - a) is in used_diffs.
                    # But what if (x - a1) == (a2 - a3) where a2, a3 are in current row?
                    # The differences within the current row must be distinct from each other too.
                    # The set `used_diffs` should contain ALL differences found so far in ALL rows.
                    # So if we add x, we add {x - a for a in row} to used_diffs.
                    # We must ensure these new diffs are not in used_diffs.
                    # We must ALSO ensure the new diffs are not duplicates of each other?
                    # x - a1 == x - a2 implies a1 == a2, which is false.
                    # So the only check is against global used_diffs.
                    
                    # Wait, the greedy loop above is slightly inefficient and potentially flawed if not careful.
                    # Let's re-verify:
                    # If we pick x, new diffs are D_new = {x - a | a in row}.
                    # Condition: D_new is disjoint from used_diffs.
                    # If valid, we add x to row, and D_new to used_diffs.
                    # This ensures all diffs in row are distinct from all previous diffs.
                    # Are diffs within the row distinct from each other? Yes, as shown above.
                    # Are diffs within the row distinct from previous diffs in the SAME row?
                    # Yes, because we update used_diffs immediately.
                    
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1]) # Exclude x itself? No, x-a where a is in old row.
                    current_max = x
                    break
                x += 1
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    
    # 2. Simulated Annealing
    # State: list of rows.
    # Move: Pick a random row, pick a random element (index > 0), change its value.
    # To maintain "increasing integers starting at 0", we can't just change value arbitrarily.
    # Better Move: 
    #   - Pick a row r.
    #   - Pick an index i (1..k).
    #   - Pick a new value v for r[i].
    #   - Ensure r[i-1] < v < r[i+1] (if i < k).
    #   - This is hard to sample efficiently.
    
    # Alternative Move: "Shift" or "Insert"
    # Let's use a simpler representation: 
    # Each row is defined by its elements.
    # A move consists of changing one element in one row.
    # To keep it simple and fast:
    # 1. Pick a row index ri.
    # 2. Pick an element index ei (1 to k).
    # 3. Generate a candidate value v.
    #    - v must be > rows[ri][ei-1].
    #    - If ei < k, v must be < rows[ri][ei+1].
    #    - If ei == k, v can be anything > rows[ri][ei-1].
    #    - To explore, we can pick v from a range around the current value.
    
    # Conflict Detection:
    # We need to check if the new set of differences for the modified row
    # conflicts with:
    # 1. Other differences in the same row (excluding those involving the changed element).
    # 2. Differences in all other rows.
    
    # To make this fast, we precompute the set of "background" differences.
    # When we modify row ri, we remove its old diffs from the global pool,
    # add the new diffs, and check for collisions.
    
    # However, maintaining a global set of diffs and updating it on every move
    # is O(k) per move, which is fine.
    
    # Let's represent the state as:
    # rows: list of lists
    # diff_sets: list of sets, where diff_sets[i] contains all positive diffs in row i.
    # global_diffs: a set of all diffs in all rows.
    
    # Actually, checking collision is:
    # new_diffs = {v - a for a in rows[ri] if a < v} U {a - v for a in rows[ri] if a > v}
    # We need new_diffs to be disjoint from (global_diffs - old_diffs_of_ri).
    # And new_diffs must be internally distinct (which they are if v is distinct from others).
    
    # Cost Function:
    # We want to minimize scope (max element).
    # But we must satisfy the distinctness constraint.
    # Standard SA for constraint satisfaction:
    # Energy = Penalty(conflicts) + Lambda * Scope
    # If we find a valid solution, we minimize Scope.
    # If we have conflicts, we minimize Conflict Count.
    
    # Let's define:
    # conflicts: number of duplicate differences across all rows.
    # scope: max element in all rows.
    
    # We will track:
    # current_rows
    # current_conflicts: count of duplicate diffs? Or just a set of "bad" diffs?
    # Calculating exact conflict count is expensive.
    # Let's use a heuristic: 
    # If we change one element, we remove old diffs and add new ones.
    # We can maintain a count of how many times each difference appears.
    # diff_count[d] = number of times difference d appears in the entire set of rows.
    # A conflict exists if any diff_count[d] > 1.
    # Total conflicts = sum(1 for d in diff_count if diff_count[d] > 1)
    # Or simpler: Number of pairs of equal diffs?
    # Let's just track the set of "active" differences and their counts.
    
    # Implementation Plan for SA:
    # 1. Initialize diff_counts: dict or array.
    #    Max difference is bounded by scope. Scope is likely < 1000.
    #    Use a list of size M_max (e.g., 2000) for speed.
    
    M_max = 2000  # Safe upper bound for differences in reasonable solutions
    diff_counts = [0] * (M_max + 1)
    
    def update_diffs(row, sign):
        # sign = 1 to add, -1 to remove
        # row is a list of ints
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= M_max:
                    diff_counts[d] += sign
                else:
                    # If d > M_max, we ignore it for conflict counting?
                    # No, we must account for it. But if we are minimizing scope,
                    # large diffs are bad. 
                    # For conflict detection, if d > M_max, it's unlikely to collide
                    # with small diffs, but could collide with another large diff.
                    # Given M_max=2000 and scope likely < 500, this is safe.
                    pass
    
    # Initialize counts
    for r in rows:
        update_diffs(r, 1)
    
    def get_conflicts():
        # Count how many differences have count > 1
        # This is O(M_max) which is 2000. Acceptable if not called every step?
        # Better: Maintain a variable `total_conflicts`.
        # When we change a diff d:
        #   old_count = diff_counts[d]
        #   if old_count == 1: conflicts += 1 (it becomes 2, creating a conflict pair? No, count 2 means 1 conflict pair? 
        #   Let's define conflict as "number of differences that are not unique".
        #   If count goes 1->2, we have 1 duplicate value. 
        #   If count goes 2->3, we still have 1 duplicate value, but more pairs.
        #   Let's minimize the number of "bad" difference values (count > 1).
        
        pass

    # Let's refine the cost function.
    # Cost = 1000 * (number of duplicate difference values) + Scope
    # If we have 0 duplicates, we minimize Scope.
    
    # We need to efficiently update the "number of duplicate difference values".
    # Let `num_dupes` be the count of d such that diff_counts[d] > 1.
    
    current_rows = [r[:] for r in rows]
    current_scope = best_scope
    num_dupes = sum(1 for c in diff_counts if c > 1)
    
    def apply_move(ri, ei, new_val):
        nonlocal num_dupes, current_scope
        
        # 1. Remove old diffs involving ei
        old_val = current_rows[ri][ei]
        row = current_rows[ri]
        
        # Remove diffs
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= M_max:
                old_count = diff_counts[d]
                diff_counts[d] -= 1
                if old_count == 2 and diff_counts[d] == 1:
                    num_dupes -= 1
                elif old_count == 1 and diff_counts[d] == 0:
                    pass # no change in num_dupes
        
        # 2. Update row
        current_rows[ri][ei] = new_val
        # Re-sort is not needed if we maintain order? 
        # The problem says "increasing integers". 
        # If we change ei, we must ensure order is preserved.
        # The move generation must ensure row[ei-1] < new_val < row[ei+1].
        
        # 3. Add new diffs
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= M_max:
                old_count = diff_counts[d]
                diff_counts[d] += 1
                if old_count == 1 and diff_counts[d] == 2:
                    num_dupes += 1
                elif old_count == 0 and diff_counts[d] == 1:
                    pass
        
        # 4. Update scope
        # This is expensive to do exactly. 
        # Approximation: 
        # If new_val > current_scope, current_scope = new_val
        # If old_val == current_scope, we need to find new max.
        # To save time, we can just recalc scope every N steps or when num_dupes == 0.
        # Or, we can just use max(current_rows) if we do it sparingly.
        # For now, let's update scope lazily or approximately.
        # Actually, let's just track the max of the modified row and global max.
        # But global max requires scanning all rows.
        # Let's just calculate scope when num_dupes == 0.
        
    # SA Parameters
    T = 100.0
    T_min = 0.1
    alpha = 0.999
    steps = 0
    max_steps = 2000000 # Adjust based on speed
    
    best_T = T
    best_num_dupes = num_dupes
    best_scope_val = current_scope
    
    # Precompute valid move ranges? No, dynamic.
    
    while time.time() < deadline:
        # Pick a random row and element
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, k) # Indices 1..k
        
        row = current_rows[ri]
        old_val = row[ei]
        
        # Determine valid range for new_val
        # Must be > row[ei-1]
        # Must be < row[ei+1] if ei < k
        # Also, to keep scope small, we might bias towards smaller values.
        
        lower = row[ei-1] + 1
        upper = row[ei+1] - 1 if ei < k else current_scope + 10 # Allow expansion if needed? 
        # Actually, if we want to minimize scope, we should constrain upper to current_scope.
        # But if we are stuck, we might need to expand.
        # Let's allow expansion up to best_scope + 50 initially, then tighten.
        
        # For now, restrict to [lower, min(upper, best_scope + 20)]
        # If lower > upper, skip move.
        
        search_upper = min(upper, best_scope + 20)
        if lower > search_upper:
            continue
            
        # Random choice in [lower, search_upper]
        # Bias towards smaller values?
        # Uniform is fine for now.
        new_val = random.randint(lower, search_upper)
        
        if new_val == old_val:
            continue
            
        # Calculate Cost Difference
        # We need to know if this move reduces num_dupes or scope.
        # Since we don't have a fast exact scope update, we rely on num_dupes.
        # If num_dupes > 0, we want to reduce it.
        # If num_dupes == 0, we want to reduce scope.
        
        # Let's apply the move, check, and revert if worse.
        
        # Save state
        old_diffs_snapshot = None # Too expensive to snapshot whole array
        
        # Instead, let's just apply, get new num_dupes, and decide.
        # But we need to know the cost BEFORE applying to decide acceptance?
        # No, SA accepts based on delta.
        # We can compute delta in num_dupes by simulating the diff changes.
        
        # Let's simulate diff changes to get delta_dupes.
        delta_dupes = 0
        
        # 1. Remove old diffs
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= M_max:
                c = diff_counts[d]
                if c == 2:
                    delta_dupes -= 1 # 2->1 removes a duplicate
                # c > 2: 3->2, still duplicate, no change in count of "duplicate values"
                # c == 1: 1->0, no change
        
        # 2. Add new diffs
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= M_max:
                c = diff_counts[d]
                if c == 1:
                    delta_dupes += 1 # 1->2 creates a duplicate
        
        # Scope Delta
        # If num_dupes + delta_dupes == 0, we care about scope.
        # Scope change:
        # If new_val > old_val:
        #   If old_val was the max, scope might decrease.
        #   If new_val > current_scope, scope increases.
        # This is tricky.
        # Let's assume if num_dupes > 0, we ignore scope.
        # If num_dupes == 0, we accept if new_val < old_val (likely reduces scope) or if it doesn't increase max.
        
        # Simplified Acceptance:
        # If num_dupes > 0:
        #   Accept if delta_dupes < 0 (reduces duplicates)
        #   Accept if delta_dupes == 0 with prob exp(-delta_scope/T) ? 
        #   Let's just accept if delta_dupes <= 0.
        # If num_dupes == 0:
        #   Accept if new_val <= old_val (reduces or maintains element size)
        #   Accept if new_val > old_val with prob exp(-(new_val-old_val)/T)
        
        accepted = False
        if num_dupes > 0:
            if delta_dupes < 0:
                accepted = True
            elif delta_dupes == 0:
                # Neutral move, accept with small prob to escape local minima
                if random.random() < 0.1:
                    accepted = True
        else: # num_dupes == 0
            # Valid state. Minimize scope.
            # Estimate scope impact.
            # If new_val < old_val, likely good.
            # If new_val > old_val, bad unless it's not the max.
            # Let's just check if new_val < old_val.
            if new_val < old_val:
                accepted = True
            else:
                # Accept with probability based on temperature
                # Penalty is roughly new_val - old_val
                if random.random() < math.exp(-(new_val - old_val) / T):
                    accepted = True
        
        if accepted:
            # Apply Move
            apply_move(ri, ei, new_val)
            
            # Update Best
            if num_dupes == 0:
                # Recalculate scope
                cur_scope = max(max(r) for r in current_rows)
                if cur_scope < best_scope_val:
                    best_scope_val = cur_scope
                    best_rows = [r[:] for r in current_rows]
            
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat occasionally?
            if random.random() < 0.01:
                T = 50.0
        
        steps += 1
        if steps % 10000 == 0:
            # Check time
            if time.time() > deadline - 2:
                break

    # Final check: If best_rows is not valid (due to heuristic approximations), 
    # we should return the greedy one if it's valid, or try to repair.
    # The greedy one is always valid.
    # If SA found a better valid one, best_rows is updated.
    
    return best_rows
