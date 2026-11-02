import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _generate_candidate_rows(n_candidates, k, max_val, n_rows, existing_rows_diffs):
    """
    Generate a pool of candidate rows that are valid Sidon rows.
    We want rows with small max values.
    """
    candidates = []
    # Use a simple randomized greedy to generate diverse candidates
    for _ in range(n_candidates):
        row = [0]
        used_local = set()
        # Try to keep values low
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                break
            new_diffs = {x - a for a in row}
            if not new_diffs & used_local:
                row.append(x)
                used_local |= new_diffs
        
        if len(row) == k + 1:
            # Check if this row is "good" (low max)
            # We don't check against existing rows here, that's done during insertion
            candidates.append(row)
            
    # Sort by max entry
    candidates.sort(key=lambda r: r[-1])
    return candidates

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing with Row Reconstruction
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = 300.0
    T_min = 0.3
    cool = 0.99997
    
    no_improve = 0
    max_no_improve = 25000
    steps = 0
    
    # Precompute row diffs for each row
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    # Generate initial pool of candidate rows for reconstruction
    # We want candidates that are smaller than the current best to allow improvement
    # But also some that are similar to allow shuffling
    # Let's generate a pool up to best_val + 10
    candidate_pool = _generate_candidate_rows(200, k, best_val + 20, n_rows, all_used)
    
    while time.time() < deadline and no_improve < max_no_improve:
        steps += 1
        
        # With some probability, attempt a Row Reconstruction
        # This is a larger move that can escape local minima
        if random.random() < 0.15:
            # Pick a row to replace, biased towards those with high max
            sorted_indices = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
            ri = random.choice(sorted_indices[:max(1, n_rows // 2)])
            
            old_row = current_rows[ri]
            old_diffs = row_diffs[ri]
            old_max = row_maxes[ri]
            
            # Find the diffs used by other rows
            other_diffs = all_used - old_diffs
            
            # Try to find a candidate row that is compatible
            # We prefer candidates with max <= current global max
            # To speed up, we only check the first few candidates (smallest max)
            found = False
            # Shuffle candidates slightly to avoid biasing always to the exact same one
            # But keep the sort order mostly
            candidates_to_try = candidate_pool[:50]
            random.shuffle(candidates_to_try)
            
            for cand in candidates_to_try:
                cand_max = cand[-1]
                # If candidate max is much larger than current row max, it might not help
                # unless it allows other rows to shrink, but we are replacing one row.
                # Ideally cand_max <= current_val (global max)
                # If cand_max > current_val, it's only good if it allows a better arrangement later?
                # For now, prefer cand_max <= current_val.
                
                cand_diffs = _row_diffs_set(cand)
                
                # Check conflict with other rows
                if cand_diffs & other_diffs:
                    continue
                
                # If no conflict, this is a valid move
                # Calculate delta
                # New global max is max(max of other rows, cand_max)
                other_maxes = [row_maxes[i] for i in range(n_rows) if i != ri]
                new_global_max = max([cand_max] + other_maxes)
                delta = new_global_max - current_val
                
                # Acceptance criteria
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    # Accept neutral moves to shuffle
                    if random.random() < 0.2:
                        accept = True
                else:
                    # Accept worse moves with prob
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Update state
                    all_used -= old_diffs
                    all_used |= cand_diffs
                    current_rows[ri] = cand[:]
                    row_diffs[ri] = cand_diffs
                    row_maxes[ri] = cand_max
                    current_val = new_global_max
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    found = True
                    break
            
            if not found:
                no_improve += 1
            
            T = max(T_min, T * cool)
            continue # Skip standard SA move this iteration
            
        # --- Standard SA Move ---
        
        # Pick a row and position to modify
        # Bias toward rows with larger max entries
        if random.random() < 0.4:
            # Pick from top half of rows by max
            sorted_indices = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
            ri = random.choice(sorted_indices[:max(1, n_rows // 2)])
        else:
            ri = random.randrange(n_rows)
        
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates - try more of these
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(80, old_val - min_allowed)
        # Try all decreases but sample if too many
        if max_decrease > 0:
            if max_decrease <= 30:
                candidates.extend(old_val - d for d in range(1, max_decrease + 1))
            else:
                # Sample a subset
                n_sample = 30
                sample_ds = random.sample(range(1, max_decrease + 1), n_sample)
                candidates.extend(old_val - d for d in sample_ds)
            
        # Increasing candidates (for exploration) - fewer
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            new_v = old_val + d
            candidates.append(new_v)
            
        # Random jump (large perturbation)
        if random.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            # Basic validity check for this position
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Build candidate row
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Calculate new diffs for this row efficiently
            old_diffs = row_diffs[ri]
            
            # Compute new diffs involving pi
            new_diffs_involving_pi = set()
            for i in range(len(cand_row)):
                if i != pi:
                    d = abs(cand_row[pi] - cand_row[i])
                    # Check internal conflict: d must not be in unchanged diffs
                    # We need to know unchanged diffs.
                    # For k=6, this is small.
                    pass
            
            # Efficient internal check:
            # Unchanged diffs are those from pairs (i,j) where i,j != pi
            unchanged_diffs = set()
            for i in range(len(cand_row)):
                if i == pi:
                    continue
                for j in range(i + 1, len(cand_row)):
                    if j == pi:
                        continue
                    unchanged_diffs.add(cand_row[j] - cand_row[i])
            
            valid_internal = True
            for i in range(len(cand_row)):
                if i == pi:
                    continue
                d = abs(cand_row[pi] - cand_row[i])
                if d in unchanged_diffs:
                    valid_internal = False
                    break
            
            if not valid_internal:
                continue
            
            # Compute full new diffs set for this row
            new_diffs = unchanged_diffs | {abs(cand_row[pi] - cand_row[i]) for i in range(len(cand_row)) if i != pi}
            
            # Check for conflicts with other rows
            # Other rows diffs = all_used - old_diffs
            # Conflict if new_diffs intersects (all_used - old_diffs)
            # Which is equivalent to: (new_diffs & all_used) - old_diffs is empty?
            # No, new_diffs might contain diffs that were in old_diffs (unchanged ones).
            # The changed diffs are new_diffs - old_diffs? No, old_diffs is the set for the OLD row.
            # The new row has new_diffs.
            # The other rows have all_used - old_diffs.
            # Conflict if new_diffs & (all_used - old_diffs) is not empty.
            
            conflict = False
            other_diffs = all_used - old_diffs
            for d in new_diffs:
                if d in other_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Move is valid
            cand_row_max = max(cand_row)
            
            # Update row_maxes temporarily
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max  # restore
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.4:  # Allow more neutral moves to shuffle
                    accept = True
            else:
                # Accept worse moves with probability
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Periodic restart if stuck
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
            # Regenerate candidate pool based on new best
            candidate_pool = _generate_candidate_rows(200, k, best_val + 20, n_rows, all_used)
    
    # 3. Targeted shrink phase: focus on the row(s) with the maximum entry
    polish_deadline = min(deadline, start_time + 118.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Identify rows that achieve the global max
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    max_no_improve_polish = 30000
    
    # Regenerate candidate pool for polish
    candidate_pool = _generate_candidate_rows(300, k, best_val + 10, n_rows, all_used)
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Bias toward rows with the maximum entry
        if max_row_indices and random.random() < 0.8:
            ri = random.choice(max_row_indices)
        else:
            # Pick from top 3 rows by max
            sorted_indices = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
            ri = random.choice(sorted_indices[:min(3, n_rows)])
        
        # With some probability, try row reconstruction in polish phase too
        if random.random() < 0.2:
            old_row = current_rows[ri]
            old_diffs = row_diffs[ri]
            other_diffs = all_used - old_diffs
            
            found = False
            for cand in candidate_pool[:50]:
                cand_diffs = _row_diffs_set(cand)
                if cand_diffs & other_diffs:
                    continue
                
                cand_max = cand[-1]
                other_maxes = [row_maxes[i] for i in range(n_rows) if i != ri]
                new_global_max = max([cand_max] + other_maxes)
                delta = new_global_max - current_val
                
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.1
                else:
                    accept = False
                
                if accept:
                    all_used -= old_diffs
                    all_used |= cand_diffs
                    current_rows[ri] = cand[:]
                    row_diffs[ri] = cand_diffs
                    row_maxes[ri] = cand_max
                    current_val = new_global_max
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                        max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                    else:
                        no_improve += 1
                    
                    found = True
                    break
            
            if not found:
                no_improve += 1
            continue
            
        # Standard polish move (decreasing last element)
        if ri in max_row_indices and random.random() < 0.6:
            pi = k  # the last element
        elif random.random() < 0.3:
            pi = random.choice(range(max(1, k - 2), k + 1))  # last few positions
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try from largest decrease to smallest (most aggressive first)
        candidates = []
        for d in range(max_decrease, 0, -1):
            candidates.append(old_val - d)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            old_diffs = row_diffs[ri]
            
            # Internal check
            unchanged_diffs = set()
            for i in range(len(cand_row)):
                if i == pi:
                    continue
                for j in range(i + 1, len(cand_row)):
                    if j == pi:
                        continue
                    unchanged_diffs.add(cand_row[j] - cand_row[i])
            
            valid_internal = True
            for i in range(len(cand_row)):
                if i == pi:
                    continue
                d = abs(cand_row[pi] - cand_row[i])
                if d in unchanged_diffs:
                    valid_internal = False
                    break
            
            if not valid_internal:
                continue
            
            new_diffs = unchanged_diffs | {abs(cand_row[pi] - cand_row[i]) for i in range(len(cand_row)) if i != pi}
            
            other_diffs = all_used - old_diffs
            conflict = False
            for d in new_diffs:
                if d in other_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            # In polish phase, only accept improving or neutral moves
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.15
            else:
                accept = False
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    # Update max_row_indices
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        # Restart from best if stuck in polish
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
            no_improve = 0
            candidate_pool = _generate_candidate_rows(300, k, best_val + 10, n_rows, all_used)
    
    # 4. Final quick SA pass with very low temperature
    final_deadline = deadline
    T = 0.2
    T_min = 0.01
    no_improve = 0
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        # Bias toward max rows
        sorted_indices = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
        if random.random() < 0.6:
            ri = random.choice(sorted_indices[:min(2, n_rows)])
        else:
            ri = random.randrange(n_rows)
        
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        # Try a random subset of decreasing moves
        n_try = min(max_decrease, random.randint(5, 12))
        candidates = random.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            old_diffs = row_diffs[ri]
            
            # Internal check
            unchanged_diffs = set()
            for i in range(len(cand_row)):
                if i == pi:
                    continue
                for j in range(i + 1, len(cand_row)):
                    if j == pi:
                        continue
                    unchanged_diffs.add(cand_row[j] - cand_row[i])
            
            valid_internal = True
            for i in range(len(cand_row)):
                if i == pi:
                    continue
                d = abs(cand_row[pi] - cand_row[i])
                if d in unchanged_diffs:
                    valid_internal = False
                    break
            
            if not valid_internal:
                continue
            
            new_diffs = unchanged_diffs | {abs(cand_row[pi] - cand_row[i]) for i in range(len(cand_row)) if i != pi}
            
            other_diffs = all_used - old_diffs
            conflict = False
            for d in new_diffs:
                if d in other_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.05
            else:
                accept = False
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    
                moved = True
                break
    
    return best_rows

# EVOLVE-BLOCK-END
