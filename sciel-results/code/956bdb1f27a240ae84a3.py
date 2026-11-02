import time
import random
import sys
import numpy as np

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
    
    # Try a few greedy starts with different randomizations
    n_starts = 5
    start_budget = min(5, deadline - time.time())
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        sol = _greedy_construct(n_rows, k, random_seed=s * 12345 + 42)
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
    no_improvement_count = 0
    while time.time() < deadline - 5:
        # Perturb the best solution using VALID moves
        candidate = [r[:] for r in best_solution]
        
        # Perform a few random valid moves (perturbation)
        n_perturb = random.randint(2, 5)
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
                no_improvement_count = 0
            elif curr_score == best_score and random.random() < 0.3:
                # Occasionally accept equal solutions to escape local minima
                best_solution = improved
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        
        # If stuck for a while, try a more aggressive shake
        if no_improvement_count > 10 and time.time() < deadline - 20:
            candidate = [r[:] for r in best_solution]
            for _ in range(8):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != candidate[r_idx][pos]:
                    candidate[r_idx][pos] = new_val
            improved = _local_search_minimize(candidate, n_rows, k, deadline)
            if improved:
                curr_score = max(max(r) for r in improved)
                if curr_score < best_score:
                    best_score = curr_score
                    best_solution = improved
                    no_improvement_count = 0
    
    # Final polish: try to squeeze out more improvement
    if time.time() < deadline - 2:
        final = _local_search_minimize([r[:] for r in best_solution], n_rows, k, deadline)
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
    
    # Track current max
    curr_max = max(max(r) for r in best_rows)
    no_progress_count = 0
    
    while time.time() < deadline - 3:
        progress = False
        
        # Focus on high values first to reduce scope quickly
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        # Sort by value descending
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Skip if value is already small
            if old_val <= curr_max - 3:
                continue

            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                diffs_involving.add(abs(row[i] - old_val))
            
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
                        d = abs(row[i] - new_val)
                        
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
                        if old_val == curr_max:
                            curr_max = max(max(r) for r in best_rows)
                        break
                
                if found:
                    no_progress_count = 0
                    continue
        
        if not progress:
            no_progress_count += 1
            if no_progress_count > 2:
                # Shake: try to move elements to different positions to escape local minimum
                # Use a more targeted approach: pick elements near the max and try various new values
                shake_attempts = 0
                for _ in range(5):
                    if shake_attempts >= 3:
                        break
                    
                    # Pick an element with high value
                    candidates = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)
                                  if best_rows[r_idx][pos] > curr_max - 10]
                    if not candidates:
                        candidates = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
                    
                    r_idx, pos = random.choice(candidates)
                    row = best_rows[r_idx]
                    old_val = row[pos]
                    
                    diffs_involving = set()
                    for i in range(n):
                        if i == pos: continue
                        diffs_involving.add(abs(row[i] - old_val))
                    forbidden_diffs = all_diffs - diffs_involving
                    
                    lower_bound = row[pos-1] + 1
                    upper_bound = min(250, row[pos+1] - 1) if pos + 1 < n else 250
                    
                    # Try multiple random values
                    tried = set()
                    for _ in range(100):
                        new_val = random.randint(lower_bound, upper_bound)
                        if new_val == old_val or new_val in tried:
                            continue
                        tried.add(new_val)
                        valid = True
                        new_diffs = set()
                        for i in range(n):
                            if i == pos: continue
                            d = abs(row[i] - new_val)
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            all_diffs.difference_update(diffs_involving)
                            all_diffs.update(new_diffs)
                            best_rows[r_idx][pos] = new_val
                            shake_attempts += 1
                            break
            else:
                break
                
    # Final polish: try to reduce each element as much as possible
    if time.time() < deadline - 1:
        # Go through all elements in order of value and try to reduce them
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 1:
                break
            
            row = best_rows[r_idx]
            old_val = row[pos]
            
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                diffs_involving.add(abs(row[i] - old_val))
            
            forbidden_diffs = all_diffs - diffs_involving
            
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        d = abs(row[i] - new_val)
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        break
                
    return best_rows

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
