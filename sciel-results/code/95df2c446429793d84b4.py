import time
import sys
import random
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # 1. Initial Greedy Construction
    best_solution = _greedy_construct(n_rows, k, rng)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # 2. Iterated Local Search
    current_solution = [r[:] for r in best_solution]
    
    while time.time() < deadline - 10:
        # Polish current solution
        polished = _local_search(current_solution, n_rows, k, deadline)
        
        curr_score = max(max(r) for r in polished)
        
        if curr_score < best_score:
            best_score = curr_score
            best_solution = [r[:] for r in polished]
            current_solution = [r[:] for r in polished]
            
            # If we've hit the theoretical lower bound or a very good score, we might stop or continue polishing
            # The counting bound is 105. If we get close, we should keep polishing.
            
        else:
            # Perturb the best solution to escape local minima
            # Replace 1 or 2 random rows with new random valid rows
            current_solution = [r[:] for r in best_solution]
            n_perturb = rng.randint(1, 2)
            for _ in range(n_perturb):
                if not current_solution:
                    break
                row_idx = rng.randrange(n_rows)
                # Generate a random valid row that fits within current best_score + margin
                # We need to ensure it doesn't conflict with other rows
                other_rows = [r for i, r in enumerate(current_solution) if i != row_idx]
                forbidden_diffs = set()
                for r in other_rows:
                    forbidden_diffs |= _row_diffs_set(r)
                
                # Try to generate a row with max <= best_score
                # If fails, try slightly larger
                new_row = _random_valid_row(forbidden_diffs, best_score, k, rng)
                if new_row is None:
                    new_row = _random_valid_row(forbidden_diffs, best_score + 20, k, rng)
                
                if new_row:
                    current_solution[row_idx] = new_row
                else:
                    # If we can't even generate a row, break perturbation
                    break

    return best_solution

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, rng):
    """
    Greedy construction of initial solution.
    Tries to build rows with small maximums.
    """
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        # We want to keep the row as small as possible.
        # Greedy: pick the smallest valid next number.
        while len(row) < k + 1:
            found = False
            # Start searching from current_max + 1
            # Limit search to a reasonable range to avoid infinite loops
            # A safe upper bound for the next element is current_max + 2*k + 5
            # But we can be tighter.
            for x in range(current_max + 1, current_max + 100):
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
                # If we couldn't find a small x, try a wider random search
                # This should rarely happen with good parameters
                for _retry in range(20):
                    x = current_max + 1 + rng.randint(0, 50)
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
        
        # Optional: Randomize the order of construction or the choice of x slightly
        # to get different initial solutions.
        # For now, deterministic greedy is a good baseline.
        
    return rows

def _random_valid_row(forbidden_diffs, max_limit, k, rng):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        # Search for candidates in a window
        # We want to keep the row compact, so we don't jump too far
        # But we must respect max_limit
        start_x = last_val + 1
        end_x = min(max_limit, last_val + 50)
        
        if start_x > max_limit:
            return None
            
        for x in range(start_x, end_x + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
                
        if not candidates:
            # Try a wider search if no candidates found in tight window
            # This allows escaping local traps in row construction
            for x in range(start_x, max_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                    candidates.append(x)
                    if len(candidates) > 10:
                        break
            
        if not candidates:
            return None
        
        # Pick a random candidate, biased towards smaller values to keep scope small
        # Use a weighted choice or just pick the smallest few
        # Let's pick from the smallest 5 candidates to encourage compactness
        candidates.sort()
        top_candidates = candidates[:min(5, len(candidates))]
        chosen_x = rng.choice(top_candidates)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _local_search(rows, n_rows, k, deadline):
    """
    Local search to minimize the maximum entry.
    Strategy: Repeatedly try to decrease the largest element in the largest row.
    Then try to decrease other elements.
    """
    n = k + 1
    current_rows = [r[:] for r in rows]
    
    # Calculate current global max
    global_max = max(max(r) for r in current_rows)
    
    # Maintain a set of all used differences for fast conflict checking
    all_diffs = set()
    for r in current_rows:
        all_diffs |= _row_diffs_set(r)
        
    # We will iterate until no improvements are made or time is up
    # To avoid getting stuck in a single local minimum, we can do multiple passes
    # or use a "no-improve" counter.
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        improved = False
        
        # Identify the row with the largest max element
        # We will focus on reducing this row first
        max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        
        # Try to reduce the last element of the max row
        # Then try to reduce other elements
        # We iterate through positions in descending order to maintain increasing property
        # Actually, we can try any position, but we must ensure row[i-1] < row[i] < row[i+1]
        
        # Let's try to reduce the largest element in the largest row
        row = current_rows[max_row_idx]
        
        # Try to reduce the last element
        pos = n - 1
        old_val = row[pos]
        lower_bound = row[pos-1] + 1
        upper_bound = old_val - 1
        
        # Collect diffs involving (max_row_idx, pos)
        diffs_involving = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                diffs_involving.add(old_val - row[i])
            else:
                diffs_involving.add(row[i] - old_val)
                
        forbidden_diffs = all_diffs - diffs_involving
        
        # Try to find a smaller valid value
        found = False
        for new_val in range(lower_bound, upper_bound + 1):
            valid = True
            new_diffs = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    d = new_val - row[i]
                else:
                    d = row[i] - new_val
                    
                if d in forbidden_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Update state
                all_diffs.difference_update(diffs_involving)
                all_diffs.update(new_diffs)
                row[pos] = new_val
                improved = True
                found = True
                break
        
        if not found:
            # If we couldn't reduce the last element, try to reduce other elements in this row
            # Or try to reduce elements in other rows that are close to the global max
            # For now, let's try to reduce any element in any row that is > global_max - 10
            # This is a heuristic to keep the search focused.
            
            # Let's just iterate through all rows and positions in a random order
            # to see if any other reduction is possible.
            indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, n)]
            random.shuffle(indices)
            
            for r_idx, pos in indices:
                if time.time() > deadline - 5:
                    break
                
                row = current_rows[r_idx]
                old_val = row[pos]
                
                # If this value is not part of the "top" values, skip
                # This optimization might miss some improvements, but speeds up search
                # if old_val < global_max - 5:
                #     continue
                
                lower_bound = row[pos-1] + 1
                upper_bound = old_val - 1
                if pos + 1 < n:
                    upper_bound = min(upper_bound, row[pos+1] - 1)
                    
                if upper_bound >= lower_bound:
                    diffs_involving = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving.add(old_val - row[i])
                        else:
                            diffs_involving.add(row[i] - old_val)
                            
                    forbidden_diffs = all_diffs - diffs_involving
                    
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        new_diffs = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val
                                
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if valid:
                            all_diffs.difference_update(diffs_involving)
                            all_diffs.update(new_diffs)
                            row[pos] = new_val
                            improved = True
                            break
                            
                if improved:
                    # Update global max if necessary
                    # We don't need to update global_max immediately, it's just a heuristic
                    break
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
    return current_rows
