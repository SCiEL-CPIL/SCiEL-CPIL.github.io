import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            for a in row:
                if (x - a) in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
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
    that do NOT involve position pi.
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
    using the precomputed static diffs plus the k diffs involving pi.
    Returns (new_diffs_set, valid_internal).
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

def _check_conflict_fast(new_diffs, old_diffs, all_used):
    """
    Check if new_diffs conflicts with all_used - old_diffs.
    Only need to check the k new diffs (those not in old_diffs).
    """
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return True
    return False

def _randomized_greedy_row(k, all_used, max_val_hint=200):
    """
    Generate a single row using randomized greedy.
    Returns (row, diffs_set) or (None, None) if fails.
    """
    row = [0]
    # We need to pick k more values.
    # To make it "randomized", we don't just pick the smallest.
    # We pick from a window of valid small values.
    
    # To ensure it's not too big, we cap the search for each new element.
    current_max = 0
    for step in range(k):
        candidates = []
        # Search for valid candidates in a small range
        # Start from current_max + 1
        # We want to keep values small, so we look at a limited window
        window_size = 50 + step * 10 
        start_val = current_max + 1
        end_val = start_val + window_size
        
        found = False
        for v in range(start_val, end_val):
            # Check internal validity and global validity
            valid = True
            for a in row:
                d = v - a
                if d in all_used:
                    valid = False
                    break
                # Check if d conflicts with other diffs in this new row?
                # We are building the row sequentially, so we must check against
                # diffs already formed in 'row' (which are in 'all_used' if we added them, 
                # but we haven't added the new row's diffs to 'all_used' yet in this function logic).
                # Wait, 'all_used' contains diffs from OTHER rows.
                # We need to check against diffs within the current partial row.
                # Let's maintain a local set 'local_diffs'.
            # Actually, it's safer to check against a local set of diffs for the partial row.
            pass
            
        # Let's refactor: maintain local_diffs
        # Restart logic properly
        
    # Simpler implementation for randomized greedy row:
    row = [0]
    local_diffs = set()
    for step in range(k):
        candidates = []
        # Try values starting from min possible
        min_val = row[-1] + 1
        # We want to explore a bit, so we look up to min_val + 20
        max_search = min_val + 30
        for v in range(min_val, max_search):
            # Check against local_diffs and all_used
            ok = True
            for a in row:
                d = v - a
                if d in local_diffs or d in all_used:
                    ok = False
                    break
            if ok:
                candidates.append(v)
                
        if not candidates:
            return None, None
            
        # Pick randomly from the top few smallest candidates to balance exploration/exploitation
        # Sort is already done by range
        if len(candidates) > 5:
            # Pick from the first 5 with some randomness
            v = random.choice(candidates[:5])
        else:
            v = random.choice(candidates)
            
        row.append(v)
        for a in row[:-1]:
            local_diffs.add(v - a)
            
    return row, local_diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # ILS Loop
    while time.time() < deadline:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used(current_rows)
        row_maxes = [max(r) for r in current_rows]
        
        # Phase 1: Local Search (SA)
        # Run SA for a fixed number of steps or time slice
        T = 50.0
        T_min = 0.5
        cool = 0.9999
        
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k) for r in current_rows]
        
        steps_in_phase = 0
        max_steps_phase = 5000 # Limit steps per phase to allow multiple ILS iterations
        
        while time.time() < deadline and steps_in_phase < max_steps_phase:
            # Pick row and position
            # Bias towards rows with high max
            row_weights = [row_maxes[r] for r in range(n_rows)]
            total_w = sum(row_weights)
            if total_w > 0:
                cum_w = []
                s = 0
                for w in row_weights:
                    s += w
                    cum_w.append(s)
                r_val = random.random() * s
                ri = 0
                while ri < n_rows - 1 and cum_w[ri] < r_val:
                    ri += 1
            else:
                ri = random.randrange(n_rows)
                
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(20, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
                
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
            max_increase = min(10, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
                
            if random.random() < 0.05:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
                if low <= high:
                    candidates.append(random.randint(low, high))
                    
            if not candidates:
                steps_in_phase += 1
                T = max(T_min, T * cool)
                continue
                
            random.shuffle(candidates)
            
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
                
                if _check_conflict_fast(new_diffs, old_diffs, all_used):
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
            
            if not moved:
                pass
                
            T = max(T_min, T * cool)
            steps_in_phase += 1
            
        # Phase 2: Perturbation (Structural Change)
        # If we still have time, perturb the current best (or current state if it improved)
        # We use the "best_rows" found so far as the base for perturbation
        # Actually, standard ILS perturbs the *current* solution which is the result of the local search.
        # Since we update best_rows inside the loop, we should perturb best_rows or the current state?
        # Let's perturb the state resulting from the last local search (which is either best_rows or slightly worse).
        # To be safe, we perturb the current state, and if it leads to a better result in the next phase, we keep it.
        
        if time.time() < deadline - 5:
            # Pick a random row to reconstruct
            ri = random.randrange(n_rows)
            
            # Remove row ri from all_used
            old_row = current_rows[ri]
            old_diffs = _row_diffs_set(old_row)
            all_used -= old_diffs
            
            # Reconstruct row ri using randomized greedy
            new_row, new_diffs = _randomized_greedy_row(k, all_used)
            
            if new_row is not None:
                # Check if this new row is valid with the rest (it should be, by construction of greedy)
                # But we must ensure it doesn't conflict with itself (handled in greedy)
                # And it doesn't conflict with others (handled by checking all_used in greedy)
                
                # Add new diffs to all_used
                all_used |= new_diffs
                
                # Update current state
                current_rows[ri] = new_row
                new_row_max = max(new_row)
                row_maxes[ri] = new_row_max
                current_val = max(row_maxes)
                
                # If this perturbation immediately yields a better solution, update best
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                else:
                    # We accept the perturbed state for the next local search phase
                    # This is the "Iterated" part: we move to a new region of the space
                    pass
            else:
                # Failed to reconstruct, restore old row
                all_used |= old_diffs
                # State remains as it was before perturbation attempt

    return best_rows

# EVOLVE-BLOCK-END
