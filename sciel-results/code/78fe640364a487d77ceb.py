import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Total differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: we need at least total_diffs distinct positive differences
    # If max entry is M, differences are in [1, M], so M >= total_diffs
    lower_bound = total_diffs
    
    # Best known for (4,7) is 134. Lower bound is 4 * 36 = 144? 
    # Wait, C(8,2) = 28. 4 * 28 = 112.
    # The prompt says "simple counting bound is 112".
    # Best known is 134.
    
    # Strategy:
    # 1. Generate a strong initial solution using a modified greedy with randomization and local search.
    # 2. Try to reduce the scope by searching for solutions with smaller max entries.
    # 3. Use a constraint-satisfaction approach with randomized restarts and backtracking.
    
    def greedy_construct(n_rows, k, scope=None, randomize=False):
        """
        Greedily construct rows. If scope is given, try to keep entries within scope.
        If randomize, shuffle candidate order.
        """
        if scope is None:
            # Estimate scope
            scope = int(total_diffs * 1.2) + 10
        
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                candidates = []
                # We need to pick x > last such that all diffs x - a (a in row) are not in used
                # and are <= scope (if we care about scope, but greedy just picks smallest valid)
                # Actually, to minimize max entry, we should pick the smallest valid x.
                # But sometimes picking a slightly larger x allows better packing later?
                # For now, smallest valid x is a good heuristic.
                
                # Optimization: only check up to a certain limit if we are trying to fit in scope?
                # No, greedy finds a solution. We want it to be as small as possible.
                # So we iterate x from last+1 upwards.
                
                # To speed up, we can pre-compute or just loop.
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        # Also check if new_diffs are distinct from each other?
                        # Yes, they are distinct because a is distinct and x is fixed.
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
                    # If x gets too large, we might have a problem, but greedy usually works.
                    # If scope is provided and x > scope, we fail?
                    # Let's just return None if we exceed a very large bound to avoid infinite loops in bad cases.
                    if x > scope + 100:
                        return None
            rows.append(row)
        
        max_entry = max(max(r) for r in rows)
        if scope is not None and max_entry > scope:
            return None
        return rows

    def check_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def local_search(rows, max_iterations=1000, time_limit=10.0):
        """
        Try to reduce the max entry by local search.
        Move: pick a row, pick an element, change it to a smaller value (if valid).
        Or: swap elements? No, must be increasing.
        So we can only decrease values.
        """
        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in best_rows)
        start = time.time()
        
        # Pre-compute used diffs for quick check?
        # It's complex to maintain dynamically during search.
        # Simple approach: try to decrease the largest element first.
        
        for _ in range(max_iterations):
            if time.time() - start > time_limit:
                break
            if best_scope <= lower_bound:
                break
            
            # Find the row with the largest element
            max_row_idx = max(range(len(best_rows)), key=lambda i: best_rows[i][-1])
            row = best_rows[max_row_idx]
            
            # Try to reduce the last element
            # Find the largest element in this row that can be reduced
            # Actually, reducing an earlier element might allow reducing later ones too.
            # Let's try to reduce the last element of the row with the max.
            
            last_val = row[-1]
            # Try smaller values for row[-1]
            found = False
            for new_val in range(last_val - 1, row[-2], -1):
                # Check if new_val is valid
                # Diffs involving new_val: new_val - row[i] for i < -1
                # Must not conflict with existing used diffs (excluding those from this row involving old last_val)
                # This is tricky. We need to remove old diffs, check, add new diffs.
                
                # Let's just do a full validity check on a candidate row.
                candidate_row = row[:-1] + [new_val]
                # Check if candidate_row is strictly increasing
                if not all(candidate_row[i] < candidate_row[i+1] for i in range(len(candidate_row)-1)):
                    continue
                    
                # Check global validity with this row replaced
                valid = True
                used_set = set()
                for idx, r in enumerate(best_rows):
                    if idx == max_row_idx:
                        r = candidate_row
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d in used_set:
                                valid = False
                                break
                            used_set.add(d)
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rows[max_row_idx] = candidate_row
                    best_scope = max(max(r) for r in best_rows)
                    found = True
                    break
            
            if not found:
                # Try other rows? Or try reducing other elements in the same row?
                # For simplicity, if we can't reduce the last element of the max row,
                # we might be stuck. A more powerful search would be needed.
                # Let's try to reduce the last element of any row.
                any_reduced = False
                for r_idx in range(len(best_rows)):
                    if any_reduced:
                        break
                    r = best_rows[r_idx]
                    last_val = r[-1]
                    for new_val in range(last_val - 1, r[-2], -1):
                        candidate_row = r[:-1] + [new_val]
                        if not all(candidate_row[i] < candidate_row[i+1] for i in range(len(candidate_row)-1)):
                            continue
                        
                        valid = True
                        used_set = set()
                        for idx, rr in enumerate(best_rows):
                            if idx == r_idx:
                                rr = candidate_row
                            for i in range(len(rr)):
                                for j in range(i + 1, len(rr)):
                                    d = rr[j] - rr[i]
                                    if d in used_set:
                                        valid = False
                                        break
                                    used_set.add(d)
                                if not valid:
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            best_rows[r_idx] = candidate_row
                            best_scope = max(max(r) for r in best_rows)
                            any_reduced = True
                            break
                if not any_reduced:
                    break
        
        return best_rows

    # 1. Get initial solution
    rows = greedy_construct(n_rows, k)
    if rows is None:
        # Fallback: very large scope
        rows = greedy_construct(n_rows, k, scope=1000)
    
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Local search to improve
    best_rows = local_search(best_rows, max_iterations=2000, time_limit=20.0)
    best_scope = max(max(r) for r in best_rows)
    
    # 3. Try to find a better solution by restarting with different randomizations or scopes
    # We can try to construct with a target scope slightly smaller than current best_scope
    # and use backtracking if greedy fails.
    
    def backtracking_search(n_rows, k, target_scope, deadline, node_limit=50000):
        """
        Search for a solution where all entries <= target_scope.
        Returns list of rows or None.
        """
        used = set()
        rows = []
        node_count = 0
        
        def backtrack(row_idx, current_row):
            nonlocal node_count
            if time.time() > deadline or node_count > node_limit:
                return False
            node_count += 1
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last = current_row[-1]
            # Candidates: x in [last+1, target_scope]
            # We want to try smaller x first to keep scope small?
            # Or random order?
            # Smaller x first is usually better for minimizing max entry.
            
            # Optimization: if remaining space is insufficient, fail early.
            # We need k+1-len(current_row) more elements.
            # The smallest possible values are last+1, last+2, ...
            # So the last element will be at least last + (k+1-len(current_row))
            # If that exceeds target_scope, fail.
            min_last_possible = last + (k + 1 - len(current_row))
            if min_last_possible > target_scope:
                return False
            
            candidates = list(range(last + 1, target_scope + 1))
            # Randomize order?
            # If we always pick smallest, we might get stuck in a local minimum.
            # Let's shuffle to explore different structures.
            random.shuffle(candidates)
            # But bias towards smaller?
            # Let's just shuffle.
            
            for x in candidates:
                # Check validity
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Add to used
                used.update(new_diffs)
                current_row.append(x)
                
                if backtrack(row_idx, current_row):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used.discard(d)
            
            return False
        
        if backtrack(0, [0]):
            return rows
        return None

    # Try to reduce scope by searching for solutions with smaller max entry
    # Start from best_scope - 1 down to lower_bound
    current_target = best_scope - 1
    
    while current_target >= lower_bound and time.time() < deadline:
        # Try a few random restarts for this target scope
        found = False
        for attempt in range(5): # 5 restarts per scope
            if time.time() > deadline:
                break
            random.seed(attempt * 1000 + current_target) # Deterministic but varied
            sol = backtracking_search(n_rows, k, current_target, deadline, node_limit=20000)
            if sol:
                # Verify
                if check_valid(sol):
                    best_rows = sol
                    best_scope = current_target
                    found = True
                    break
        
        if found:
            # Continue to next smaller scope
            current_target -= 1
        else:
            # If we fail to find a solution for current_target,
            # it might be impossible, or just hard.
            # Given time limits, we might stop if we've failed a few times?
            # Or just decrement and try again?
            # If it's impossible, we'll waste time.
            # Let's stop if we've tried 3 scopes in a row without success?
            # For now, just decrement.
            current_target -= 1
            
            # Safety: if we've gone too low, stop
            if current_target < lower_bound:
                break
    
    # Final check
    if not check_valid(best_rows):
        # Should not happen, but fallback
        rows = greedy_construct(n_rows, k)
        if rows:
            best_rows = rows
    
    return best_rows
