import time
import sys
import random
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all differences between two entries of the same row are distinct over all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 115
    
    # Phase 1: Multi-start greedy initialization
    # We try multiple random seeds to find a good starting landscape
    best_solution = None
    best_score = float('inf')
    
    # Number of starts depends on time available, but we cap it to ensure we have time for optimization
    n_starts = 10
    start_budget = min(10, (deadline - time.time()) / 10) # Rough estimate
    
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Use a deterministic seed based on the start index to ensure reproducibility of the "multi-start" logic
        # but allow some variation if we were to randomize the seed itself. 
        # Here we use a fixed offset to explore different greedy paths.
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

    # Initial local search polish on the best start
    best_solution = _local_search_minimize_fast(best_solution, n_rows, k, deadline)
    best_score = max(max(r) for r in best_solution)

    # Phase 2: Iterated Local Search (ILS) with Targeted Perturbation
    # We perturb the current best solution, then run local search.
    # If the result is better, we accept it.
    # If it's equal or slightly worse, we might accept it with a small probability to escape local minima.
    
    stagnation_counter = 0
    max_stagnation = 50
    
    while time.time() < deadline - 15:
        candidate = [r[:] for r in best_solution]
        
        # Targeted Perturbation:
        # Focus on elements that contribute to the high score (max values)
        # We pick a few random positions, but bias towards the last element of rows
        # or elements that are close to the current max.
        
        n_perturb = random.randint(2, 5)
        for _ in range(n_perturb):
            r_idx = random.randint(0, n_rows - 1)
            # Bias towards later positions in the row, as they tend to be larger
            # and thus more likely to be the bottleneck.
            pos = random.randint(1, k)
            if random.random() < 0.5:
                # Force perturbation on the last element
                pos = k
            
            new_val = _find_random_valid_value_fast(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val
        
        # Local search to minimize scope
        improved = _local_search_minimize_fast(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
                # If we've been stuck for a while, do a stronger perturbation or restart
                if stagnation_counter > max_stagnation:
                    # Stronger perturbation: change more elements
                    candidate = [r[:] for r in best_solution]
                    for _ in range(random.randint(5, 10)):
                        r_idx = random.randint(0, n_rows - 1)
                        pos = random.randint(1, k)
                        new_val = _find_random_valid_value_fast(candidate, r_idx, pos, n_rows, k)
                        if new_val is not None and new_val != candidate[r_idx][pos]:
                            candidate[r_idx][pos] = new_val
                            
                    improved = _local_search_minimize_fast(candidate, n_rows, k, deadline)
                    if improved:
                        curr_score = max(max(r) for r in improved)
                        if curr_score < best_score:
                            best_score = curr_score
                            best_solution = improved
                            stagnation_counter = 0
                    # Reset counter to avoid infinite tight loops of strong perturbation
                    stagnation_counter = 0

    # Final polish: aggressive local search
    if time.time() < deadline - 5:
        final = _local_search_minimize_fast([r[:] for r in best_solution], n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    return best_solution

def _find_random_valid_value_fast(rows, r_idx, pos, n_rows, k):
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
    # We want to explore a range around the current value, potentially smaller to reduce scope
    lower_bound = row[pos-1] + 1
    # Upper bound: allow some expansion if needed for validity, but prefer smaller
    # In local search, we usually want to stay close or go smaller.
    # For perturbation, we can go a bit larger to escape.
    upper_bound = max(old_val + 20, row[pos-1] + 50) 
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    # Collect valid candidates
    valid_candidates = []
    # To speed up, we can sample or scan. Given small k, scanning is fine.
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
    
    # Bias towards values that reduce the maximum?
    # For perturbation, random choice is good for exploration.
    # For local search, we handle minimization separately.
    return random.choice(valid_candidates)

def _local_search_minimize_fast(rows, n_rows, k, deadline):
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
    
    progress = True
    no_improvement_count = 0
    max_no_improvement = 10 # Increased slightly to allow more thorough search per pass
    
    while progress and time.time() < deadline - 5:
        progress = False
        # Shuffle indices to randomize search order
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
            # We strictly look for values smaller than old_val to minimize scope
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                # Iterate from smallest possible to find the tightest fit
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
                    no_improvement_count = 0
                    continue
        
        if not progress:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
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
            # Try values starting from current_max + 1
            # We scan a window of potential next values
            # The window size can be randomized to break ties or explore different paths
            window_size = rng.randint(20, 50)
            
            for x in range(current_max + 1, current_max + window_size + 1):
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
                # If no small value works, try a wider search with randomization
                # This helps escape dead ends in the greedy path
                for _retry in range(100):
                    x = current_max + 1 + rng.randint(0, 100)
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
                    # Failed to construct this row, return None to signal failure
                    return None
        rows.append(row)
    return rows

# Keep old functions for compatibility if they were referenced, though they are not used in the new main loop
def _local_search_ilS(rows, n_rows, k, deadline):
    return _local_search_minimize_fast(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    return None

def _find_smallest_valid_value(rows, r_idx, pos, old_val, n_rows, k):
    return None
