import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Constants
    # The problem asks for 4 rows of 8 integers (k=7? No, k is passed).
    # Task: "Choose 4 rows of 8 increasing integers". 
    # If k=7, length is 8. If k is passed as 7, len = k+1 = 8.
    # The prompt says "Each row is a list of k + 1 integers".
    
    # 1. CSP Backtracking Initialization
    # We want to find a valid configuration minimizing scope.
    # We construct rows one by one.
    # State: rows (list of lists), used_diffs (set)
    
    def try_construct_csp(n_rows, k, max_scope_hint):
        """
        Attempts to construct a valid set of rows using backtracking.
        max_scope_hint is an upper bound for the largest element to guide search.
        """
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, current_max):
            if row_idx == n_rows:
                return True
            
            # If we are in the first row, or starting a new row
            if len(current_row) == 0:
                current_row = [0]
                current_max = 0
            
            # Try to complete the current row
            while len(current_row) < k + 1:
                # Determine candidate range for next element
                # To minimize scope, try small increments first
                # But we also need to ensure we don't exceed max_scope_hint
                start_val = current_max + 1
                end_val = min(max_scope_hint, 500) # Hard cap for sanity in search
                
                found = False
                for x in range(start_val, end_val + 1):
                    # Check if x creates a conflict with used_diffs
                    valid = True
                    new_diffs = []
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        # Check if new_diffs have internal conflicts?
                        # No, x - a1 != x - a2 for a1 != a2.
                        # So we just need to check against global used_diffs.
                        
                        # Apply move
                        current_row.append(x)
                        current_max = x
                        used_diffs.update(new_diffs)
                        
                        if len(current_row) < k + 1:
                            if backtrack(row_idx, current_row, current_max):
                                return True
                        else:
                            # Row complete
                            rows.append(current_row[:])
                            if backtrack(row_idx + 1, [], 0):
                                return True
                            
                        # Undo move
                        current_row.pop()
                        for d in new_diffs:
                            used_diffs.discard(d)
                        current_max = current_row[-1] if current_row else 0
                        found = True
                        break # If we found a valid x, we don't need to try larger x for this step? 
                              # Actually, in backtracking, if this branch fails later, we need to try next x.
                              # So we should NOT break here. We should continue the for loop if the recursive call fails.
                              # But the structure above is slightly off. Let's restructure.
                
                # The standard backtracking pattern is cleaner:
                # for x in range(...):
                #   if valid:
                #      apply
                #      if recurse: return True
                #      undo
                # return False
                
                # Let's rewrite the inner loop properly inside the function below.
                return False # This line is reached if no x worked in the current iteration of "completing the row"
                # Wait, the logic above is convoluted. Let's implement a clean helper.
            
            return False
            
        # Clean Backtracking Implementation
        def bt(row_idx, current_row, current_max):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == 0:
                current_row = [0]
                current_max = 0
                
            if len(current_row) >= k + 1:
                # Row is complete, move to next
                rows.append(current_row[:])
                return bt(row_idx + 1, [], 0)
                
            # Need to add next element to current_row
            start_val = current_max + 1
            # Bound the search space to keep it fast.
            # If max_scope_hint is provided, use it.
            # Otherwise, a dynamic bound.
            # For initialization, we might just let it grow, but we want small scope.
            # We'll try values in increasing order.
            
            # To prune, we can estimate if we can fit the remaining elements.
            # Remaining elements: k + 1 - len(current_row)
            # Min increment is 1. So min needed space = remaining.
            # current_max + remaining <= max_scope_hint
            
            max_possible_val = max_scope_hint
            for x in range(start_val, max_possible_val + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Apply
                    current_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    if bt(row_idx, current_row, x):
                        return True
                    
                    # Undo
                    current_row.pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
            
            return False
            
        # Try increasing scope hints
        # The counting bound is 112. Best known 134.
        # Let's try 134 first, then 150, 200, etc.
        for scope_hint in [134, 140, 150, 180, 250, 500]:
            rows = []
            used_diffs = set()
            if bt(0, [0], 0):
                return rows, scope_hint
        return None, None

    # 2. Local Search / ILS for Minimization
    
    # Helper to check validity and get conflicts
    def get_conflicts_and_scope(rows):
        diff_counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = sum(1 for c in diff_counts.values() if c > 1)
        scope = max(max(r) for r in rows)
        return conflicts, scope

    # Try to get initial valid solution via CSP
    init_rows, init_scope = try_construct_csp(n_rows, k, 134)
    
    if init_rows is None:
        # Fallback: Greedy construction if CSP fails or times out (it shouldn't for small k)
        used_diffs = set()
        init_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                valid = True
                for a in row:
                    if x - a in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1])
                    x += 1
                else:
                    x += 1
            init_rows.append(row)
        init_scope = max(max(r) for r in init_rows)
        
    best_rows = [r[:] for r in init_rows]
    best_scope = init_scope
    
    # ILS Loop
    # State: rows
    # Goal: Minimize scope
    # Move: Pick random element, change it, repair conflicts.
    
    def repair(rows):
        """
        Greedily fix conflicts by moving elements to smaller valid values if possible,
        or just finding any valid value to satisfy constraints, preferring small ones.
        Returns True if valid.
        """
        max_iter = 1000
        for _ in range(max_iter):
            conflicts, scope = get_conflicts_and_scope(rows)
            if conflicts == 0:
                return True, rows
            
            # Find a conflicting difference
            diff_counts = {}
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                        
            conflict_diffs = [d for d, c in diff_counts.items() if c > 1]
            if not conflict_diffs:
                return True, rows
            
            target_diff = random.choice(conflict_diffs)
            
            # Find a row and pair (i, j) with this difference
            moved = False
            for ri, r in enumerate(rows):
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        if r[j] - r[i] == target_diff:
                            # We need to change either r[i] or r[j].
                            # Changing r[j] (the larger one) is usually safer for maintaining order with subsequent elements?
                            # Or changing r[i]?
                            # Let's try changing r[j] to a smaller value if possible, or larger.
                            # Actually, we want to minimize scope, so we prefer smaller values.
                            # But we must maintain strict increase.
                            
                            # Try to change r[j] to a value v such that:
                            # r[j-1] < v < r[j+1] (if exists)
                            # And v - r[i] != target_diff (and no other conflicts)
                            
                            # This is complex. Simpler: Randomly pick one of the conflicting elements and nudge it.
                            # Let's nudge r[j].
                            old_val = r[j]
                            # Try a new value in a range around old_val
                            # Range must respect neighbors
                            lower = r[j-1] + 1 if j > 0 else 1
                            upper = r[j+1] - 1 if j < len(r)-1 else 1000 # Scope limit?
                            
                            # To minimize scope, we should try values < old_val first?
                            # If old_val is the max, reducing it is good.
                            # If old_val is not max, reducing it is good.
                            
                            found_new_val = False
                            # Try random values in [lower, min(upper, 200)]
                            # Bias towards smaller
                            for _ in range(50):
                                if lower > min(upper, 200):
                                    break
                                new_val = random.randint(lower, min(upper, 200))
                                if new_val == old_val:
                                    continue
                                
                                # Check if this move resolves the conflict and creates no new ones
                                # Simulate move
                                # This is expensive to do fully. 
                                # Let's just apply and check global validity in the outer loop?
                                # No, repair should be somewhat smart.
                                
                                # Check if new_val creates any duplicate diff in this row
                                valid_local = True
                                # Remove old diffs involving j
                                # Add new diffs
                                # Check against global set?
                                # For speed, let's just do a simple check:
                                # Is (new_val - r[i]) already in the row? Or global?
                                # We'll rely on the outer validity check if this is too slow.
                                # Let's just swap and let the outer loop reject if invalid.
                                # But 'repair' should make progress.
                                
                                # Let's use a simpler heuristic: 
                                # If we just swap, the outer ILS step will check validity.
                                # If invalid, it reverts.
                                # So 'repair' can just be "make a random valid move that reduces scope".
                                
                                # Actually, let's just pick a random valid value for r[j] that is smaller than old_val if possible.
                                r[j] = new_val
                                found_new_val = True
                                moved = True
                                break
                            if moved:
                                break
                    if moved:
                        break
                if moved:
                    break
                    
            if not moved:
                # Try expanding or randomizing
                # Pick random element and random value
                ri = random.randint(0, n_rows - 1)
                ei = random.randint(1, k)
                r = rows[ri]
                lower = r[ei-1] + 1
                upper = r[ei+1] - 1 if ei < k else best_scope + 10
                if lower <= upper:
                    new_val = random.randint(lower, upper)
                    r[ei] = new_val
                    moved = True
                if not moved:
                    return False, rows
            
        return get_conflicts_and_scope(rows)[0] == 0, rows

    # ILS Main Loop
    T = 100.0
    alpha = 0.99
    T_min = 1.0
    
    while time.time() < deadline:
        # Perturbation (Kick)
        # Pick 2-5 random elements and change them significantly
        kick_size = random.randint(2, 5)
        for _ in range(kick_size):
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            r = best_rows[ri]
            old_val = r[ei]
            lower = r[ei-1] + 1
            upper = r[ei+1] - 1 if ei < k else best_scope + 20
            if lower <= upper:
                # Pick a value far from old_val to escape local minima
                # But keep it within bounds
                # To minimize scope, maybe pick from [lower, old_val] or [old_val, upper]
                # Random uniform is fine for kick
                new_val = random.randint(lower, upper)
                r[ei] = new_val
        
        # Local Search (Repair/Refine)
        # We want to reach a valid state with minimal scope.
        # We'll run a simple local search:
        # 1. If invalid, fix conflicts.
        # 2. If valid, try to reduce scope by moving max elements down.
        
        current_rows = [r[:] for r in best_rows]
        
        # Phase 1: Fix conflicts
        for step in range(100):
            if time.time() > deadline:
                break
            conflicts, scope = get_conflicts_and_scope(current_rows)
            if conflicts == 0:
                break
            
            # Pick a random conflicting element and try to move it to a non-conflicting spot
            # Find a diff with count > 1
            diff_counts = {}
            for r in current_rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                        
            conflict_diffs = [d for d, c in diff_counts.items() if c > 1]
            if not conflict_diffs:
                break
            target_diff = random.choice(conflict_diffs)
            
            # Find one instance
            for ri in range(n_rows):
                r = current_rows[ri]
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        if r[j] - r[i] == target_diff:
                            # Try to change r[j]
                            old_val = r[j]
                            lower = r[j-1] + 1 if j > 0 else 1
                            upper = r[j+1] - 1 if j < len(r)-1 else scope + 10
                            
                            # Try a few random values
                            for _ in range(10):
                                if lower > upper: break
                                new_val = random.randint(lower, upper)
                                if new_val == old_val: continue
                                
                                # Check if this move helps
                                # Quick check: does new_val create a diff that is already in conflict?
                                # Just apply and check global conflicts in next loop iteration?
                                # Or check locally:
                                valid = True
                                # Check against other elements in same row
                                for m in range(len(r)):
                                    if m == j: continue
                                    d_new = abs(r[m] - new_val)
                                    # If d_new is in diff_counts and diff_counts[d_new] > 0 (from other rows) or 
                                    # d_new is created by another pair in this row...
                                    # This is getting complicated.
                                    # Let's just apply and let the outer loop check.
                                    # But we need to avoid infinite loops of bad moves.
                                    pass
                                
                                # Simplest: Apply, check if conflicts decreased.
                                r[j] = new_val
                                new_conflicts, _ = get_conflicts_and_scope(current_rows)
                                if new_conflicts < conflicts:
                                    break # Keep move
                                else:
                                    r[j] = old_val # Revert
                            break
                    break
                break
            
        # Phase 2: Minimize Scope (if valid)
        conflicts, scope = get_conflicts_and_scope(current_rows)
        if conflicts == 0:
            # Try to reduce the maximum element
            for step in range(50):
                if time.time() > deadline:
                    break
                # Find the max element
                max_val = max(max(r) for r in current_rows)
                # Find its location
                target_ri, target_ei = -1, -1
                for ri, r in enumerate(current_rows):
                    if max(r) == max_val:
                        target_ri = ri
                        target_ei = r.index(max_val)
                        break
                
                if target_ri == -1: break
                
                r = current_rows[target_ri]
                old_val = r[target_ei]
                lower = r[target_ei-1] + 1 if target_ei > 0 else 1
                # Upper bound is old_val - 1 (to reduce scope)
                # But must be > lower
                upper = old_val - 1
                
                if lower <= upper:
                    # Try random values in [lower, upper]
                    # Prefer values close to old_val? Or random?
                    # Random to escape local minima
                    for _ in range(10):
                        new_val = random.randint(lower, upper)
                        r[target_ei] = new_val
                        new_conflicts, new_scope = get_conflicts_and_scope(current_rows)
                        if new_conflicts == 0 and new_scope < scope:
                            scope = new_scope
                            break
                        else:
                            r[target_ei] = old_val # Revert if not better or invalid
                else:
                    break # Can't reduce max further in this position
            
            # If we found a valid solution with lower scope, update best
            if conflicts == 0 and scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in current_rows]
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat occasionally
            if random.random() < 0.05:
                T = 50.0

    return best_rows

# Helper for testing if needed
if __name__ == "__main__":
    rows = construct(4, 7)
    print(rows)
    print("Scope:", max(max(r) for r in rows))
