import time
import random
import math

# EVOLVE-BLOCK-START

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

def _greedy_randomized(n_rows, k, max_val=None):
    """
    Randomized greedy: at each step, pick from among the smallest valid candidates
    with some randomness to get diverse solutions.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Find several valid candidates
            valid_candidates = []
            search_limit = 20
            for _ in range(search_limit):
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    valid_candidates.append(x)
                    if len(valid_candidates) >= 5:
                        break
            if not valid_candidates:
                # Fallback: keep searching
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(x)
                        break
            # Pick randomly from candidates (bias toward smaller)
            weights = [1.0 / (c - row[-1] + 1) for c in valid_candidates]
            total = sum(weights)
            rand_val = random.random() * total
            cumsum = 0
            chosen = valid_candidates[-1]
            for c, w in zip(valid_candidates, weights):
                cumsum += w
                if rand_val <= cumsum:
                    chosen = c
                    break
            new_diffs = {chosen - a for a in row}
            row.append(chosen)
            used |= new_diffs
        rows.append(row)
    return rows

def _sa_search(rows, n_rows, k, start_time, deadline, T_init=150.0):
    """
    Run simulated annealing on the given rows, returning the best found.
    """
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    T = T_init
    T_min = 0.5
    cool = 0.99998
    
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline:
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        # Bias toward last position (which is the max of the row)
        if random.random() < 0.5:
            pi = k
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            new_v = old_val - d
            candidates.append(new_v)
            
        # Increasing candidates (for exploration)
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            new_v = old_val + d
            candidates.append(new_v)
            
        # Random jump
        if random.random() < 0.2:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
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
            
            # Check for conflicts
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Compute new global max
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
                if random.random() < 0.3:
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
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Restart if stuck
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = T_init
            no_improve = 0
            
            # Apply random perturbations
            for _ in range(3):
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else old_val + 8
                if min_allowed < max_allowed:
                    new_val = random.randint(min_allowed, max_allowed)
                    static_pi = _build_static_diffs(current_rows[ri], k)[pi]
                    new_diffs, valid_internal = _compute_new_diffs_fast(
                        current_rows[ri], pi, new_val, static_pi)
                    if valid_internal:
                        old_diffs = _row_diffs_set(current_rows[ri])
                        all_used_check = _compute_all_used(current_rows)
                        all_used_check -= old_diffs
                        if not (new_diffs & all_used_check):
                            current_rows[ri][pi] = new_val
                            all_used = _compute_all_used(current_rows)
                            row_diffs = [_row_diffs_set(r) for r in current_rows]
                            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
                            row_maxes = [max(r) for r in current_rows]
                            current_val = _global_max(current_rows)
                            break
    
    return best_rows, best_val

def _polish(rows, n_rows, k, deadline):
    """
    Final exhaustive polish: try all possible decreases for each position.
    """
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            for pi in range(k, 0, -1):
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                static_pi = static_diffs[ri][pi]
                old_diffs = row_diffs[ri]
                
                for new_val in range(min_allowed, old_val):
                    if pi < k and new_val >= current_rows[ri][pi + 1]:
                        continue
                    
                    new_diffs, valid_internal = _compute_new_diffs_fast(
                        current_rows[ri], pi, new_val, static_pi)
                    
                    if not valid_internal:
                        continue
                    
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
                    
                    if new_global_max < current_val:
                        all_used -= old_diffs
                        all_used |= new_diffs
                        current_rows[ri][pi] = new_val
                        row_diffs[ri] = new_diffs
                        static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                        row_maxes[ri] = cand_row_max
                        current_val = new_global_max
                        improved = True
                        break
                    elif new_global_max == current_val and random.random() < 0.5:
                        all_used -= old_diffs
                        all_used |= new_diffs
                        current_rows[ri][pi] = new_val
                        row_diffs[ri] = new_diffs
                        static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                        row_maxes[ri] = cand_row_max
                        current_val = new_global_max
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return current_rows, current_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Phase 1: Generate diverse initial solutions
    best_rows = None
    best_val = float('inf')
    
    # Standard greedy
    rows1 = _greedy_init(n_rows, k)
    val1 = _global_max(rows1)
    best_rows = [r[:] for r in rows1]
    best_val = val1
    
    # Randomized greedy variants
    for _ in range(5):
        rows_r = _greedy_randomized(n_rows, k)
        val_r = _global_max(rows_r)
        if val_r < best_val:
            best_val = val_r
            best_rows = [r[:] for r in rows_r]
    
    # Phase 2: Run SA on multiple starting points
    remaining_time = deadline - time.time()
    sa_time_per = remaining_time / 4.0
    
    candidates = [best_rows]
    # Add some diverse candidates
    for _ in range(3):
        rows_r = _greedy_randomized(n_rows, k)
        candidates.append(rows_r)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        sa_deadline = min(deadline, time.time() + sa_time_per)
        sa_rows, sa_val = _sa_search(cand, n_rows, k, start_time, sa_deadline)
        if sa_val < best_val:
            best_val = sa_val
            best_rows = [r[:] for r in sa_rows]
    
    # Phase 3: Final polish
    polish_deadline = min(deadline, time.time() + 20.0)
    final_rows, final_val = _polish(best_rows, n_rows, k, polish_deadline)
    if final_val < best_val:
        best_val = final_val
        best_rows = [r[:] for r in final_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
