import time
import sys
import random
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    # Phase 0: Initial greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Phase 1: Aggressive local search to minimize scope
    while time.time() < deadline - 30:
        improved = _local_search_minimize_fast(best_solution, n_rows, k, deadline)
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
            else:
                break
    
    # Phase 2: Simulated Annealing with row re-solve
    rows = [r[:] for r in best_solution]
    all_diffs = [_row_diffs_set(r) for r in rows]
    current_max = max(r[-1] for r in rows)
    best_max = best_score
    best_rows = [r[:] for r in rows]
    
    T = 15.0
    cooling_rate = 0.995
    
    while time.time() < deadline - 15:
        i = random.randint(0, n_rows - 1)
        
        # Calculate forbidden diffs for row i
        forbidden = set()
        for j in range(n_rows):
            if j != i:
                forbidden |= all_diffs[j]
        
        # Try to find a better row using randomized greedy
        # First try to find one with max < current_max
        new_row = _randomized_greedy_row(forbidden, k, max_limit=current_max, max_candidates=15)
        
        if new_row is None:
            # If we couldn't find one under current max, try to find any valid row
            new_row = _randomized_greedy_row(forbidden, k, max_limit=None, max_candidates=15)
        
        if new_row is None:
            T *= cooling_rate
            continue
            
        # Compute new global max if we replace row i
        old_row = rows[i]
        new_global_max = 0
        for j in range(n_rows):
            if j == i:
                new_global_max = max(new_global_max, new_row[-1])
            else:
                new_global_max = max(new_global_max, rows[j][-1])
        
        delta = new_global_max - current_max
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0.01:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
            
        if accept:
            rows[i] = new_row
            all_diffs[i] = _row_diffs_set(new_row)
            current_max = new_global_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
                best_solution = [r[:] for r in rows]
                best_score = best_max
        else:
            # Reject: revert
            pass
            
        T *= cooling_rate
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 3:
        final = _local_search_minimize_fast(best_solution, n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    return best_solution

def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i+1, n):
            s.add(row[j] - row[i])
    return s

def _randomized_greedy_row(forbidden_diffs, k, max_limit=None, max_candidates=5):
    """Build a single row greedily with randomization.
    forbidden_diffs: set of differences that cannot be used.
    max_limit: if set, the last element must be < max_limit.
    max_candidates: how many valid candidates to consider at each step.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        x = last_val + 1
        
        # Safety bound for search
        search_limit = max_limit if max_limit is not None else 500
        if search_limit is None:
            search_limit = last_val + 300
        else:
            search_limit = min(search_limit, last_val + 300)
        
        while x <= search_limit:
            if max_limit is not None and x >= max_limit:
                break
                
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row) and len(new_diffs) == len(row):
                candidates.append(x)
                
                if len(candidates) >= max_candidates:
                    break
                
            x += 1
        
        if not candidates:
            return None
        
        # Pick randomly from candidates, biased towards smaller values
        # Use weighted random choice with exponential decay
        weights = [math.exp(-c / 50.0) for c in candidates]
        total_w = sum(weights)
        r = random.random() * total_w
        cumulative = 0
        chosen_x = candidates[0]
        for c, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                chosen_x = c
                break
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _local_search_minimize_fast(rows, n_rows, k, deadline):
    # Fast local search with incremental diff tracking
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    progress = True
    no_improve_count = 0
    
    while progress and time.time() < deadline - 5:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 5:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
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
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        no_improve_count = 0
                        break
                
                if not found:
                    no_improve_count += 1
    
    return best_rows

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try values starting from current_max + 1
            for x in range(current_max + 1, current_max + 500):
                new_diffs = []
                valid = True
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
                # Try a wider search with some randomization
                for _retry in range(50):
                    x = current_max + 1 + random.randint(0, 100)
                    new_diffs = []
                    valid = True
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

def _find_random_valid_value_fast(rows, r_idx, pos, n_rows, k):
    # Find a random valid value for rows[r_idx][pos] using precomputed diff sets
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)
    
    # Compute all diffs in the system
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    # Diffs involving old_val at position pos
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    # Determine search range
    lower_bound = row[pos-1] + 1
    upper_bound = 300
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        
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
            valid_candidates.append(new_val)
    
    if not valid_candidates:
        return None
    
    return random.choice(valid_candidates)

def _local_search_ilS(rows, n_rows, k, deadline):
    # Deprecated - use _local_search_minimize_fast instead
    return _local_search_minimize_fast(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    # Placeholder for repair functionality
    return None

def _find_smallest_valid_value(rows, r_idx, pos, old_val, n_rows, k):
    # Kept for compatibility but not used in the fast path
    row = rows[r_idx]
    n = len(row)
    
    all_diffs = set()
    for r in rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    lower_bound = row[pos-1] + 1
    upper_bound = old_val - 1
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    if upper_bound < lower_bound:
        return None
    
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
            return new_val
    
    return None
