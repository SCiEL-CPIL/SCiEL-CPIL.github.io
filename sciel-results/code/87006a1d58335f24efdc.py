import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all differences between two entries of the same row are distinct over all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 110
    
    # Phase 1: Multi-start greedy initialization
    best_solution = None
    best_score = float('inf')
    
    # Try several greedy starts with different randomizations
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 40:
            break
        sol = _greedy_construct(n_rows, k, random_seed=s * 7919 + 13)
        if sol:
            score = max(max(r) for r in sol)
            if score < best_score:
                best_score = score
                best_solution = sol
    
    if best_solution is None:
        # Fallback: very simple greedy
        best_solution = _greedy_construct(n_rows, k, random_seed=0)
        if not best_solution:
            return []
        best_score = max(max(r) for r in best_solution)
    
    # Phase 2: Iterated Local Search with valid perturbation
    stall_count = 0
    max_stall = 15
    
    while time.time() < deadline - 5:
        # Perturb the best solution using VALID moves
        candidate = [r[:] for r in best_solution]
        
        # Perform random valid moves (perturbation)
        n_perturb = random.randint(2, 6)
        for _ in range(n_perturb):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val
        
        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                stall_count = 0
            elif curr_score == best_score and random.random() < 0.3:
                # Occasionally accept equal solutions to escape local minima
                best_solution = improved
                stall_count += 1
            else:
                stall_count += 1
        
        # If stalled, try row reconstruction
        if stall_count >= max_stall and time.time() < deadline - 10:
            reconstructed = _try_row_reconstruction(best_solution, n_rows, k, deadline)
            if reconstructed:
                recon_score = max(max(r) for r in reconstructed)
                if recon_score < best_score:
                    best_score = recon_score
                    best_solution = reconstructed
                    stall_count = 0
                else:
                    # Keep the reconstructed version even if not better, to escape local minimum
                    if random.random() < 0.5:
                        best_solution = reconstructed
                        best_score = recon_score
                    stall_count = 0
    
    # Final polish: try to squeeze out more improvement
    if time.time() < deadline - 2:
        final = _final_polish([r[:] for r in best_solution], n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    return best_solution

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """
    Find a random valid value for rows[r_idx][pos] that maintains all-diff-distinct property.
    Returns None if no valid value found in a reasonable range.
    """
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
    upper_bound = 250
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

def _try_row_reconstruction(best_solution, n_rows, k, deadline):
    """
    Try to reconstruct one or more rows to escape local minima.
    Rebuild a row from scratch while keeping other rows fixed.
    """
    if time.time() > deadline - 15:
        return None
    
    n = k + 1
    best_sol = [r[:] for r in best_solution]
    
    # Find the row with the largest max value
    max_row_idx = 0
    max_val = -1
    for i, r in enumerate(best_sol):
        if max(r) > max_val:
            max_val = max(r)
            max_row_idx = i
    
    # Compute all diffs from other rows
    other_diffs = set()
    for i, r in enumerate(best_sol):
        if i == max_row_idx:
            continue
        for a in range(n):
            for b in range(a+1, n):
                other_diffs.add(r[b] - r[a])
    
    # Try to rebuild row max_row_idx
    new_row = _build_single_row(other_diffs, n, deadline)
    if new_row:
        best_sol[max_row_idx] = new_row
        return best_sol
    
    return None

def _build_single_row(forbidden_diffs, n, deadline):
    """
    Build a single row of n increasing integers starting at 0,
    avoiding all differences in forbidden_diffs.
    """
    row = [0]
    used_diffs = set()
    
    while len(row) < n:
        if time.time() > deadline - 10:
            return None
        
        found = False
        current_max = row[-1]
        base = current_max + 1
        
        # Try sequential values
        for x in range(base, base + 200):
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in forbidden_diffs or d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                row.append(x)
                used_diffs.update(new_diffs)
                found = True
                break
        
        if not found:
            return None
    
    return row

def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Local search that tries to reduce the maximum value by moving elements to smaller positions.
    Uses incremental diff tracking for speed.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Track which diffs are "hot" (involved in the max value)
    max_val = max(max(r) for r in best_rows)
    
    # Precompute row lengths for speed
    while time.time() < deadline - 3:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
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
                        break
                
                if found:
                    continue
        
        if not progress:
            break
            
    return best_rows

def _final_polish(rows, n_rows, k, deadline):
    """
    High-precision final polish: try multiple passes with tighter constraints.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    best_score = max(max(r) for r in best_rows)
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Multiple polish passes with decreasing tolerance
    for pass_num in range(3):
        if time.time() > deadline - 1:
            break
        
        # Try to reduce each element to its minimum valid value
        for r_idx in range(n_rows):
            for pos in range(k, 0, -1):  # Work backwards from largest to smallest
                if time.time() > deadline - 1:
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
                
                # Find the smallest valid value
                lower_bound = row[pos-1] + 1
                upper_bound = old_val - 1
                if pos + 1 < n:
                    upper_bound = min(upper_bound, row[pos+1] - 1)
                
                if upper_bound >= lower_bound:
                    best_new_val = None
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
                            best_new_val = new_val
                            break
                    
                    if best_new_val is not None and best_new_val < old_val:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = best_new_val
    
    # Check if we improved
    new_score = max(max(r) for r in best_rows)
    if new_score < best_score:
        return best_rows
    
    return rows

def _greedy_construct(n_rows, k, random_seed=0):
    """
    Greedy construction: build rows one by one, choosing smallest possible values.
    Uses randomization to explore different starting points.
    """
    rng = random.Random(random_seed)
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try values starting from current_max + 1, with some randomization
            base = current_max + 1
            search_range = min(200, 300 - current_max)
            
            # First try sequential
            for x in range(base, base + search_range):
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
                # Try randomized search
                for _retry in range(100):
                    x = base + rng.randint(0, 100)
                    if x > 250:
                        continue
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

def _local_search_ilS(rows, n_rows, k, deadline):
    # Deprecated - use _local_search_minimize instead
    return _local_search_minimize(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    # Placeholder for compatibility
    return None
