"""Difference triangle set (7,5) solver with Simulated Annealing / Iterated Local Search."""
import time
import random
import sys

def construct(n_rows, k):
    # Safety margin: 150s kill, use 110s for search
    deadline = time.time() + 110.0
    
    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    
    def get_diffs(row):
        """Compute all differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def randomized_greedy(n_rows, k, deadline):
        """
        Generate a random valid set of rows using greedy with randomization.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            success = True
            for _ in range(k):
                if time.time() > deadline:
                    success = False
                    break
                
                # Find valid candidates for the next element
                # We look for x > current_max
                # To introduce randomness, we don't just take the first one.
                # We collect a few valid candidates and pick one.
                
                # First, find the smallest valid x to ensure we don't get stuck
                smallest_valid = None
                valid_candidates = []
                
                # Search range: current_max + 1 to current_max + 100 (or more if needed)
                # If we can't find one in 100, we expand.
                limit = current_max + 100
                x = current_max + 1
                while x <= limit:
                    # Check if x is valid
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        if smallest_valid is None:
                            smallest_valid = x
                        valid_candidates.append((x, new_diffs))
                        # Collect up to 5 candidates
                        if len(valid_candidates) >= 5:
                            break
                    x += 1
                
                if not valid_candidates:
                    # If no candidates in the window, extend search greedily
                    x = limit + 1
                    while True:
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if valid:
                            valid_candidates.append((x, new_diffs))
                            break
                        x += 1
                        if x > 1000: # Safety break
                            success = False
                            break
                    if not success:
                        break

                # Choose a candidate
                # Bias towards smaller values to keep scope low, but allow some randomness
                # Weighted choice: weight = 1 / (x - current_max + 1)
                total_weight = 0
                weights = []
                for x_val, _ in valid_candidates:
                    w = 1.0 / (x_val - current_max + 1)
                    weights.append(w)
                    total_weight += w
                
                r = random.random() * total_weight
                cum = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen_idx = i
                        break
                
                x_val, new_diffs = valid_candidates[chosen_idx]
                row.append(x_val)
                used_diffs.update(new_diffs)
                current_max = x_val
            
            if success:
                rows.append(row)
            else:
                return None
        
        return rows

    def local_search(rows, deadline):
        """
        Iterated Local Search to minimize the scope (max entry).
        Moves: Change a specific element to a smaller valid value.
        Perturbations: Change an element to a different valid value (possibly larger) to escape local minima.
        """
        current_rows = [r[:] for r in rows]
        current_scope = max(max(r) for r in current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        # Precompute used_diffs for the current state
        # used_diffs_all: set of all diffs
        # row_diffs[i]: set of diffs in row i
        used_diffs_all = set()
        row_diffs = []
        for r in current_rows:
            rd = get_diffs(r)
            row_diffs.append(rd)
            used_diffs_all.update(rd)
        
        no_improve_count = 0
        max_no_improve = 50 # If we don't improve for 50 iterations, perturb
        
        while time.time() < deadline - 5:
            improved = False
            perturbed = False
            
            # 1. Try to improve: find any element that can be reduced
            # Shuffle rows and positions to avoid bias
            row_indices = list(range(n_rows))
            random.shuffle(row_indices)
            
            for ri in row_indices:
                if time.time() > deadline - 5:
                    break
                
                row = current_rows[ri]
                # Try positions from largest to smallest? Or random?
                # Reducing a larger index element often has more freedom.
                # Let's try random positions.
                pos_indices = list(range(1, len(row)))
                random.shuffle(pos_indices)
                
                for pi in pos_indices:
                    if time.time() > deadline - 5:
                        break
                    
                    old_val = row[pi]
                    
                    # We want to find a new_val < old_val
                    # Constraints:
                    # 1. row[pi-1] < new_val < row[pi+1] (if pi+1 exists)
                    # 2. All diffs involving new_val must not be in (used_diffs_all - row_diffs[ri])
                    
                    lower_bound = row[pi-1] + 1
                    upper_bound = row[pi+1] - 1 if pi + 1 < len(row) else old_val - 1
                    
                    if lower_bound > upper_bound:
                        continue
                    
                    # Calculate the set of diffs we are freeing up
                    # Diffs involving old_val in this row:
                    # old_val - row[i] for i < pi
                    # row[j] - old_val for j > pi
                    freed_diffs = set()
                    for i in range(pi):
                        freed_diffs.add(old_val - row[i])
                    for j in range(pi + 1, len(row)):
                        freed_diffs.add(row[j] - old_val)
                    
                    # The "forbidden" set is used_diffs_all - freed_diffs
                    # But we can just check against used_diffs_all and allow if in freed_diffs
                    # Actually, it's easier to compute the set of diffs *not* available:
                    # available_diffs = used_diffs_all - freed_diffs
                    # But wait, the new diffs must not be in (used_diffs_all - freed_diffs).
                    # So new_diff must NOT be in used_diffs_all OR new_diff MUST be in freed_diffs.
                    # However, new_diffs are generated from new_val and other elements in the row.
                    # Diffs between other elements in the row are already in row_diffs[ri] and thus in used_diffs_all.
                    # They are also in freed_diffs? No, freed_diffs only contains diffs involving the old_val.
                    # So diffs between other elements are still "occupied" by this row.
                    # So, for a new candidate new_val:
                    # new_diffs = {new_val - row[i] for i < pi} U {row[j] - new_val for j > pi}
                    # These new_diffs must not intersect with (used_diffs_all - freed_diffs).
                    # Equivalently: new_diffs must be a subset of freed_diffs OR disjoint from used_diffs_all.
                    # Since new_diffs are new, they are likely disjoint from used_diffs_all unless they match freed_diffs.
                    # Actually, if a new_diff matches a freed_diff, that's fine because that diff is being "replaced".
                    # But if a new_diff matches a diff from *another* row, that's a conflict.
                    # So the condition is: (new_diffs & (used_diffs_all - freed_diffs)) is empty.
                    
                    forbidden_set = used_diffs_all - freed_diffs
                    
                    # Search for new_val in [lower_bound, upper_bound)
                    # We want the smallest one to reduce scope.
                    found_val = None
                    # Iterate from lower_bound to upper_bound - 1
                    # To speed up, we can check if there's any valid value.
                    # Since the range might be small, linear scan is okay.
                    for new_val in range(lower_bound, upper_bound):
                        # Check diffs
                        valid = True
                        # Diffs with previous elements
                        for i in range(pi):
                            d = new_val - row[i]
                            if d in forbidden_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Diffs with next elements
                        for j in range(pi + 1, len(row)):
                            d = row[j] - new_val
                            if d in forbidden_set:
                                valid = False
                                break
                        if valid:
                            found_val = new_val
                            break
                    
                    if found_val is not None:
                        # Apply change
                        # Update row
                        row[pi] = found_val
                        
                        # Update used_diffs
                        # Remove old diffs
                        used_diffs_all -= freed_diffs
                        row_diffs[ri] -= freed_diffs
                        
                        # Add new diffs
                        new_freed_diffs = set() # This is the set of diffs involving new_val
                        for i in range(pi):
                            d = found_val - row[i]
                            used_diffs_all.add(d)
                            row_diffs[ri].add(d)
                        for j in range(pi + 1, len(row)):
                            d = row[j] - found_val
                            used_diffs_all.add(d)
                            row_diffs[ri].add(d)
                        
                        # Update scope
                        new_scope = max(max(r) for r in current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in current_rows]
                        
                        # If we made an improvement, restart the scan of rows? 
                        # Or continue? Continuing is fine, but restarting might find more immediate improvements.
                        # Let's just break out of the position loop and continue to next row or restart.
                        # Breaking out of position loop is sufficient.
                        break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # If stuck, try a perturbation
                if no_improve_count >= max_no_improve:
                    # Perturb: pick a random row and position, change to a random valid value
                    # This might increase scope, but hopefully allows further reduction later.
                    ri = random.randrange(n_rows)
                    row = current_rows[ri]
                    pi = random.randrange(1, len(row))
                    
                    old_val = row[pi]
                    lower_bound = row[pi-1] + 1
                    upper_bound = row[pi+1] - 1 if pi + 1 < len(row) else old_val + 50 # Allow some increase
                    
                    # Find a valid value different from old_val
                    # We look for any valid value in [lower_bound, upper_bound]
                    # To make it a perturbation, we can look for one that is not old_val.
                    
                    freed_diffs = set()
                    for i in range(pi):
                        freed_diffs.add(old_val - row[i])
                    for j in range(pi + 1, len(row)):
                        freed_diffs.add(row[j] - old_val)
                    
                    forbidden_set = used_diffs_all - freed_diffs
                    
                    # Try to find a valid new_val
                    # We'll try a few random spots in the range
                    candidates = []
                    if upper_bound > lower_bound:
                        # Sample some candidates
                        for _ in range(10):
                            cand = random.randint(lower_bound, upper_bound)
                            if cand == old_val:
                                continue
                            valid = True
                            for i in range(pi):
                                d = cand - row[i]
                                if d in forbidden_set:
                                    valid = False
                                    break
                            if not valid: continue
                            for j in range(pi + 1, len(row)):
                                d = row[j] - cand
                                if d in forbidden_set:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(cand)
                                break # Just take the first one found in random tries? 
                                # Or collect a few.
                    
                    if candidates:
                        new_val = candidates[0]
                        row[pi] = new_val
                        
                        # Update used_diffs
                        used_diffs_all -= freed_diffs
                        row_diffs[ri] -= freed_diffs
                        
                        for i in range(pi):
                            d = new_val - row[i]
                            used_diffs_all.add(d)
                            row_diffs[ri].add(d)
                        for j in range(pi + 1, len(row)):
                            d = row[j] - new_val
                            used_diffs_all.add(d)
                            row_diffs[ri].add(d)
                        
                        current_scope = max(max(r) for r in current_rows)
                        # Do NOT update best_scope here, as perturbation might worsen it
                        # But we reset no_improve_count to try to recover
                        no_improve_count = 0
                        perturbed = True

    # Main Loop
    best_solution = None
    best_score = float('inf')
    
    # Initial greedy
    init_sol = randomized_greedy(n_rows, k, deadline)
    if init_sol:
        best_solution = init_sol
        best_score = max(max(r) for r in best_solution)
    
    # Run local search on the initial solution
    if best_solution:
        improved_sol = local_search(best_solution, deadline)
        # local_search returns best found inside it? 
        # My local_search above modifies in place and tracks best. 
        # Let's return the best from local_search.
        # Wait, I need to modify local_search to return the best rows.
        
        # Let's refactor local_search to return (best_rows, best_scope)
        # Actually, I'll just call a wrapper.
        
    # Let's re-run the logic with a cleaner structure for the final code.
    
    return best_solution

# Refactored Local Search to return best found
def _local_search_minimize(rows, n_rows, k, deadline):
    current_rows = [r[:] for r in rows]
    current_scope = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    used_diffs_all = set()
    row_diffs = []
    for r in current_rows:
        rd = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                rd.add(r[j]-r[i])
        row_diffs.append(rd)
        used_diffs_all.update(rd)
        
    no_improve_count = 0
    
    while time.time() < deadline - 2:
        improved = False
        row_indices = list(range(n_rows))
        random.shuffle(row_indices)
        
        for ri in row_indices:
            if time.time() > deadline - 2:
                break
            row = current_rows[ri]
            pos_indices = list(range(1, len(row)))
            random.shuffle(pos_indices)
            
            for pi in pos_indices:
                if time.time() > deadline - 2:
                    break
                old_val = row[pi]
                lower_bound = row[pi-1] + 1
                upper_bound = row[pi+1] - 1 if pi + 1 < len(row) else old_val - 1
                
                if lower_bound > upper_bound:
                    continue
                
                freed_diffs = set()
                for i in range(pi):
                    freed_diffs.add(old_val - row[i])
                for j in range(pi + 1, len(row)):
                    freed_diffs.add(row[j] - old_val)
                
                forbidden_set = used_diffs_all - freed_diffs
                
                found_val = None
                # Search for smallest valid value
                for new_val in range(lower_bound, upper_bound):
                    valid = True
                    for i in range(pi):
                        d = new_val - row[i]
                        if d in forbidden_set:
                            valid = False
                            break
                    if not valid: continue
                    for j in range(pi + 1, len(row)):
                        d = row[j] - new_val
                        if d in forbidden_set:
                            valid = False
                            break
                    if valid:
                        found_val = new_val
                        break
                
                if found_val is not None:
                    row[pi] = found_val
                    used_diffs_all -= freed_diffs
                    row_diffs[ri] -= freed_diffs
                    for i in range(pi):
                        d = found_val - row[i]
                        used_diffs_all.add(d)
                        row_diffs[ri].add(d)
                    for j in range(pi + 1, len(row)):
                        d = row[j] - found_val
                        used_diffs_all.add(d)
                        row_diffs[ri].add(d)
                    
                    new_scope = max(max(r) for r in current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        improved = True
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in current_rows]
                    break # Move to next row after one improvement
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 20:
                # Perturb
                ri = random.randrange(n_rows)
                row = current_rows[ri]
                pi = random.randrange(1, len(row))
                old_val = row[pi]
                lower_bound = row[pi-1] + 1
                upper_bound = row[pi+1] - 1 if pi + 1 < len(row) else old_val + 20
                
                if upper_bound >= lower_bound:
                    freed_diffs = set()
                    for i in range(pi):
                        freed_diffs.add(old_val - row[i])
                    for j in range(pi + 1, len(row)):
                        freed_diffs.add(row[j] - old_val)
                    
                    forbidden_set = used_diffs_all - freed_diffs
                    
                    # Find a valid perturbation
                    # Try random candidates
                    for _ in range(5):
                        cand = random.randint(lower_bound, upper_bound)
                        if cand == old_val: continue
                        valid = True
                        for i in range(pi):
                            d = cand - row[i]
                            if d in forbidden_set:
                                valid = False
                                break
                        if not valid: continue
                        for j in range(pi + 1, len(row)):
                            d = row[j] - cand
                            if d in forbidden_set:
                                valid = False
                                break
                        if valid:
                            row[pi] = cand
                            used_diffs_all -= freed_diffs
                            row_diffs[ri] -= freed_diffs
                            for i in range(pi):
                                d = cand - row[i]
                                used_diffs_all.add(d)
                                row_diffs[ri].add(d)
                            for j in range(pi + 1, len(row)):
                                d = row[j] - cand
                                used_diffs_all.add(d)
                                row_diffs[ri].add(d)
                            break
                    no_improve_count = 0
                    
    return best_rows, best_scope

def construct(n_rows, k):
    deadline = time.time() + 110.0
    best_solution = None
    best_score = float('inf')
    
    # 1. Randomized Greedy
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        random.seed(attempt)
        sol = _randomized_greedy(n_rows, k, deadline)
        if sol:
            score = max(max(r) for r in sol)
            if score < best_score:
                best_score = score
                best_solution = sol
    
    if not best_solution:
        # Fallback
        return _greedy_construct(n_rows, k)

    # 2. Local Search / ILS
    improved_sol, improved_score = _local_search_minimize(best_solution, n_rows, k, deadline)
    if improved_score < best_score:
        best_score = improved_score
        best_solution = improved_sol
        
    return best_solution

def _randomized_greedy(n_rows, k, deadline):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        success = True
        for _ in range(k):
            if time.time() > deadline:
                success = False
                break
            valid_candidates = []
            limit = current_max + 100
            x = current_max + 1
            while x <= limit:
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 5:
                        break
                x += 1
            
            if not valid_candidates:
                x = limit + 1
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        valid_candidates.append((x, new_diffs))
                        break
                    x += 1
                    if x > 1000:
                        success = False
                        break
                if not success:
                    break
            
            total_weight = sum(1.0/(x_val - current_max + 1) for x_val, _ in valid_candidates)
            r = random.random() * total_weight
            cum = 0
            chosen_idx = 0
            for i, (x_val, _) in enumerate(valid_candidates):
                cum += 1.0/(x_val - current_max + 1)
                if r <= cum:
                    chosen_idx = i
                    break
            
            x_val, new_diffs = valid_candidates[chosen_idx]
            row.append(x_val)
            used_diffs.update(new_diffs)
            current_max = x_val
        
        if success:
            rows.append(row)
        else:
            return None
    return rows

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows
