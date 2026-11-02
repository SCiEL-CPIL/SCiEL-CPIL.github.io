import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
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

def _build_static_diffs(row, k):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi. These are differences between pairs
    (i,j) where i != pi and j != pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static

def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    """
    Given a row with position pi replaced by new_val, compute the full diff set
    using the precomputed static diffs (diffs not involving pi) plus the k diffs
    involving pi. Returns (new_diffs_set, valid_internal).
    """
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)
    return new_diffs, valid

def _randomized_greedy_row(forbidden_diffs, max_limit=None, max_candidates=5):
    """
    Build a single row greedily but with randomness.
    At each step, find up to max_candidates valid values and pick one randomly.
    This allows escaping local traps in the row structure.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        x = last_val + 1
        # Safety limit to avoid infinite loop if max_limit is None or too high
        hard_limit = max_limit if max_limit is not None else 10000
        
        while len(candidates) < max_candidates:
            if x > hard_limit:
                break
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row) and len(new_diffs) == len(row):
                candidates.append(x)
            x += 1
            # If we've scanned too far, stop
            if x - last_val > 200:
                break

        if not candidates:
            return None
        
        # Pick randomly, but bias towards smaller values slightly to keep scope tight
        # Uniform is fine, but let's try uniform first as per inspiration
        chosen_x = random.choice(candidates)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing with Row Re-solve
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    
    # Precompute row diffs and static diffs
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Choose between local move (1 element) and global move (1 row)
        # Global moves (row re-solve) are less frequent but powerful
        use_row_resolve = random.random() < 0.15
        
        if use_row_resolve:
            ri = random.randrange(n_rows)
            
            # Compute forbidden diffs: union of diffs from all other rows
            forbidden = set()
            for j in range(n_rows):
                if j != ri:
                    forbidden |= row_diffs[j]
            
            # Try to find a new row with max < current_val (improvement)
            # or just any valid row (exploration)
            # We try to constrain the max to current_val first
            new_row = _randomized_greedy_row(forbidden, max_limit=current_val, max_candidates=10)
            
            if new_row is None:
                # If we couldn't find one under current max, try without limit (might increase max, but SA can accept)
                # However, to keep scope controlled, maybe limit to current_val + 20?
                new_row = _randomized_greedy_row(forbidden, max_limit=current_val + 20, max_candidates=10)
            
            if new_row is None:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            old_diffs = row_diffs[ri]
            new_diffs = _row_diffs_set(new_row)
            
            # Check global validity (should be true by construction against forbidden, but double check internal)
            # _randomized_greedy_row ensures internal validity and no conflict with forbidden.
            
            # Update state
            old_row = current_rows[ri]
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            
            # Update all_used
            all_used -= old_diffs
            all_used |= new_diffs
            
            # Update row max
            old_row_max = row_maxes[ri]
            new_row_max = new_row[-1]
            row_maxes[ri] = new_row_max
            
            # Compute new global max
            new_global_max = max(row_maxes)
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                current_val = new_global_max
                static_diffs[ri] = _build_static_diffs(new_row, k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_rows[ri] = old_row
                row_diffs[ri] = old_diffs
                all_used -= new_diffs
                all_used |= old_diffs
                row_maxes[ri] = old_row_max
                no_improve += 1
            
        else:
            # Standard local move: modify one element
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)  # 1..k
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates with bias toward reducing max
            candidates = []
            
            # Decreasing candidates - try more of these
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(50, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                new_v = old_val - d
                candidates.append(new_v)
                
            # Increasing candidates (for exploration) - fewer
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            max_increase = min(15, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                new_v = old_val + d
                candidates.append(new_v)
                
            # Random jump (large perturbation)
            if random.random() < 0.15:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
                if low <= high:
                    candidates.append(random.randint(low, high))
                    
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            random.shuffle(candidates)
            
            # Pre-fetch static diffs for this position
            static_pi = static_diffs[ri][pi]
            
            moved = False
            for new_val in candidates:
                # Basic validity check for this position
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                    
                # Fast diff computation using static diffs
                new_diffs, valid_internal = _compute_new_diffs_fast(
                    current_rows[ri], pi, new_val, static_pi)
                
                if not valid_internal:
                    continue
                    
                # Get old diffs of row ri
                old_diffs = row_diffs[ri]
                
                # Check for conflicts: new_diffs should not intersect with (all_used - old_diffs)
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        conflict = True
                        break
                
                if conflict:
                    continue
                    
                # Move is valid - compute new global max efficiently
                if pi == k:
                    cand_row_max = new_val
                else:
                    cand_row_max = current_rows[ri][k]
                
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
                    if random.random() < 0.2:  # Allow some neutral moves to shuffle
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
                    current_rows[ri][pi] = new_val
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    
                    # Rebuild static diffs for this row
                    static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                    
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
        if no_improve > 12000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # 3. Targeted shrink phase: focus on the row(s) with the maximum entry
    polish_deadline = min(deadline, start_time + 113.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Identify rows that achieve the global max
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    max_no_improve_polish = 20000
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Bias toward rows with the maximum entry
        if max_row_indices and random.random() < 0.7:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        # For the max row, prioritize the last position (which is the max)
        if ri in max_row_indices and random.random() < 0.5:
            pi = k  # the last element
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try from largest decrease to smallest (most aggressive first)
        candidates = [old_val - d for d in range(max_decrease, 0, -1)]
        
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
            
            old_diffs = row_diffs[ri]
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            # In polish phase, only accept improving or neutral moves
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.1
            else:
                accept = False
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
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
    
    # 4. Final quick SA pass with very low temperature
    final_deadline = deadline
    T = 0.2
    T_min = 0.01
    cool = 0.99999
    no_improve = 0
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        # Try a random subset of decreasing moves
        n_try = min(max_decrease, random.randint(3, 8))
        candidates = random.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
            
            old_diffs = row_diffs[ri]
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
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
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    
                moved = True
                break
    
    return best_rows

# EVOLVE-BLOCK-END
