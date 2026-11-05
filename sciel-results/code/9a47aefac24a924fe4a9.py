import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with best-fit greedy using bitmask
    seed = 0
    max_initial_seeds = 50
    while time.time() < deadline - 45 and seed < max_initial_seeds:
        rows = _greedy_construct_bitmask(n_rows, k, seed)
        if rows is None:
            seed += 1
            continue
        scope = _scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
        seed += 1
    
    if best_rows is None:
        best_rows = _greedy_construct_bitmask(n_rows, k, 0)
        if best_rows is None:
            # Fallback to simple range if greedy fails (unlikely)
            return [[0] + list(range(1, k+1))] * n_rows
        
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated annealing to improve
    if time.time() < deadline - 10:
        improved = _simulated_annealing_bitmask(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Final polish pass targeting the maximum element
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max_bitmask(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _greedy_construct_bitmask(n_rows, k, seed):
    """
    Greedy construction using bitmask for O(1) difference conflict checks.
    We view the problem as packing 'difference shadows' into the integer line.
    """
    rng = random.Random(seed)
    used_mask = 0  # Bitmask of used differences. Bit i is set if diff i is used.
    rows = []
    
    # Estimate max difference needed. 
    # Lower bound: total_diffs distinct positive ints. 
    # Upper bound heuristic: usually around 1.2 * total_diffs / n_rows + max_row_span
    # For n=7, k=5, total_diffs = 7*15=105.
    # Let's set a generous limit for the mask. 1024 bits is enough for m~112.
    MAX_DIFF_LIMIT = 256 
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            
            # We want to find the smallest x > last_x such that 
            # {x - a for a in row} does not intersect used_mask.
            # To improve packing, we might prefer x that fills gaps?
            # For now, "First Fit" (smallest x) is a strong baseline for compactness.
            # We can add a small randomization to explore.
            
            best_x = None
            # Limit search to reasonable range
            # A row of k+1 elements with max M has max diff M.
            # If we are aiming for scope S, we shouldn't exceed S.
            # Let's search up to MAX_DIFF_LIMIT.
            
            for x in range(last_x + 1, MAX_DIFF_LIMIT):
                # Check conflicts
                # new_diffs are x - row[0], x - row[1], ...
                # We can check if (used_mask >> (x - a)) & 1 is 0 for all a in row.
                
                conflict = False
                for a in row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                
                if not conflict:
                    best_x = x
                    break
            
            if best_x is None:
                return None
            
            # Optional: Randomize choice among top few feasible x to avoid getting stuck in one pattern
            # For now, stick to First Fit for determinism and speed, but seed varies row order?
            # Actually, varying the order of rows or the starting offset might help.
            # Let's just use the found best_x.
            
            row.append(best_x)
            # Update mask
            for a in row[:-1]:
                d = best_x - a
                used_mask |= (1 << d)
        
        rows.append(row)
    
    return rows

def _simulated_annealing_bitmask(initial_rows, n_rows, k, start_time, deadline):
    """
    Simulated annealing using bitmask for fast validity checks.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute masks for current rows to speed up removal?
    # Or just recompute used_mask excluding the row being changed.
    
    T_start = 50.0
    T_end = 0.1
    total_steps = 100000 # Increase steps due to speedup
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    steps_since_improve = 0
    max_no_improve = 5000
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Select row to modify
        # Bias towards rows with larger max element
        row_idx = rng.randint(0, n_rows - 1)
        if rng.random() < 0.3:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        
        # Generate new row
        new_row = _generate_candidate_row_bitmask(rows, row_idx, k, rng)
        
        if new_row is None:
            continue
        
        # Calculate new scope
        # We don't need to check full validity of all rows if we only changed one,
        # BUT we must ensure new_row doesn't conflict with others.
        # The generation function ensures no conflict with others.
        
        new_scope = _scope([r if i != row_idx else new_row for i, r in enumerate(rows)])
        
        # Acceptance
        delta = new_scope - current_scope
        if delta <= 0:
            rows[row_idx] = new_row
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows[row_idx] = new_row
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
        
        if steps_since_improve > max_no_improve:
            # Perturb
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row_bitmask(rows, pert_idx, k, rng)
            if pert_row is not None:
                # Check validity of pert_row against others
                # It's generated to be valid against others, so it is valid.
                rows[pert_idx] = pert_row
                current_scope = _scope(rows)
            steps_since_improve = 0
            T = max(T, 1.0)
    
    return best_rows

def _generate_candidate_row_bitmask(rows, exclude_idx, k, rng):
    """
    Generate a valid row that does not conflict with other rows.
    Uses bitmask for fast conflict checking.
    """
    # Build mask of differences used by OTHER rows
    used_mask = 0
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                d = row[b] - row[a]
                used_mask |= (1 << d)
    
    row = [0]
    MAX_DIFF_LIMIT = 256
    
    for _ in range(k):
        last_x = row[-1]
        best_x = None
        
        # First Fit with some randomization
        # Collect a few feasible candidates
        candidates = []
        
        for x in range(last_x + 1, MAX_DIFF_LIMIT):
            conflict = False
            for a in row:
                d = x - a
                if (used_mask >> d) & 1:
                    conflict = True
                    break
            
            if not conflict:
                candidates.append(x)
                # Stop early if we have enough candidates?
                # Or keep looking for "better" packing?
                # For speed, First Fit is good. Let's take the first few.
                if len(candidates) >= 5:
                    break
        
        if not candidates:
            return None
        
        # Choose randomly from top candidates to explore
        chosen_x = rng.choice(candidates)
        
        row.append(chosen_x)
        # Update local mask for this row's generation
        for a in row[:-1]:
            d = chosen_x - a
            used_mask |= (1 << d)
    
    return row

def _polish_max_bitmask(rows, n_rows, k, start_time, deadline):
    """
    Try to reduce the scope by specifically targeting the row with the max element.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    for iteration in range(50):
        if time.time() > deadline - 3:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to find a row with max < current_max
        improved_row = _reduce_max_row_bitmask(rows, max_idx, k, rng, current_max)
        
        if improved_row is not None and improved_row[-1] < current_max:
            new_rows = rows[:]
            new_rows[max_idx] = improved_row
            # Validity is guaranteed by construction
            new_scope = _scope(new_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in new_rows]
                rows = new_rows
            else:
                # If scope didn't improve (maybe other rows have same max?), 
                # but max_idx row improved, it might help future steps?
                # Actually, if scope is max, and we reduced the unique max, scope should drop.
                # If multiple rows have same max, reducing one doesn't drop scope.
                pass
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row_bitmask(rows, row_idx, k, rng, current_max):
    """
    Try to generate a row with max element strictly less than current_max.
    """
    used_mask = 0
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                d = row[b] - row[a]
                used_mask |= (1 << d)
    
    # Target max is current_max - 1
    target_max = current_max - 1
    
    row = [0]
    success = True
    
    for _ in range(k):
        last_x = row[-1]
        best_x = None
        
        # Must be <= target_max
        for x in range(last_x + 1, target_max + 1):
            conflict = False
            for a in row:
                d = x - a
                if (used_mask >> d) & 1:
                    conflict = True
                    break
            
            if not conflict:
                best_x = x
                break
        
        if best_x is None:
            success = False
            break
        
        row.append(best_x)
        for a in row[:-1]:
            d = best_x - a
            used_mask |= (1 << d)
    
    if success:
        return row
    return None

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)
