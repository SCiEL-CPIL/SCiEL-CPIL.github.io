import time
import random
import math

# EVOLVE-BLOCK-START

def _diffs_set(row):
    """Compute the set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

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
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _global_max(rows):
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

def _randomized_greedy_row(k, forbidden_diffs, max_limit, max_candidates=5):
    """
    Builds a row of k+1 elements starting at 0, strictly increasing,
    such that no difference is in forbidden_diffs and all are <= max_limit.
    At each step, it finds the first `max_candidates` valid integers and picks one at random.
    Returns None if no such row can be constructed.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        x = last_val + 1
        limit_check = max_limit + 100 if max_limit is not None else 10000
        while x < limit_check:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
                if len(candidates) >= max_candidates:
                    break
            x += 1
            
        if not candidates:
            return None
        
        chosen_x = random.choice(candidates)
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _backtracking_row(k, forbidden_diffs, max_limit, max_time=0.5):
    """
    Backtracking search for a row of k+1 elements starting at 0, strictly increasing,
    such that no difference is in forbidden_diffs and all are <= max_limit.
    Uses iterative deepening on the max value.
    Returns None if no such row can be constructed within max_time.
    """
    start_time = time.time()
    
    # First, find the minimum possible max value using greedy
    min_max = 0
    row = [0]
    used_in_row = set()
    last_val = 0
    for _ in range(k):
        x = last_val + 1
        found = False
        while x <= max_limit + 100:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                row.append(x)
                used_in_row |= new_diffs
                last_val = x
                found = True
                break
            x += 1
        if not found:
            return None
    min_max = last_val
    
    # Now do backtracking search for a row with max <= min_max (or slightly higher)
    for target_max in range(min_max, min(max_limit, min_max + 10) + 1):
        if time.time() - start_time > max_time:
            break
        
        # Try to find a row with max exactly target_max using backtracking
        row = [0]
        used_in_row = set()
        
        def backtrack(pos, current_max):
            if time.time() - start_time > max_time:
                return False
            if pos == k + 1:
                return current_max <= target_max
            
            start = row[-1] + 1
            end = min(target_max, current_max) if pos < k else target_max
            
            # Try values from smallest to largest
            for x in range(start, end + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden_diffs or new_diffs & used_in_row:
                    continue
                
                row.append(x)
                new_used = set(used_in_row)
                new_used |= new_diffs
                
                if pos == k:
                    # This is the last element, check if max is within target
                    if x <= target_max:
                        return True
                    else:
                        row.pop()
                        continue
                else:
                    if backtrack(pos + 1, max(current_max, x)):
                        return True
                    row.pop()
                    new_used = used_in_row  # Restore
            
            return False
        
        if backtrack(1, 0):
            return row
    
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = set()
    for row in rows:
        all_used |= _diffs_set(row)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Track state
    current_rows = [r[:] for r in rows]
    current_val = best_val
    row_maxes = [max(r) for r in rows]
    row_diffs = [_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    # 2. Main Phase: Simulated Annealing with Row Replacement
    T = 50.0
    T_min = 0.5
    cool = 0.9998
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Decide between local move (SA) and global move (Row Replacement)
        if T > 5.0 and random.random() < 0.3:
            # --- Row Replacement Move ---
            ri = random.randrange(n_rows)
            
            # Compute forbidden diffs (all diffs except row ri)
            forbidden = set()
            for j in range(n_rows):
                if j != ri:
                    forbidden |= row_diffs[j]
            
            # Try backtracking first for better quality
            new_row = _backtracking_row(k, forbidden, best_val, max_time=0.3)
            
            if new_row is None:
                # Fall back to randomized greedy
                new_row = _randomized_greedy_row(k, forbidden, best_val, max_candidates=10)
            
            if new_row is None:
                new_row = _randomized_greedy_row(k, forbidden, current_val + 20, max_candidates=10)
            
            if new_row is not None:
                new_row_max = max(new_row)
                old_row_max = row_maxes[ri]
                
                row_maxes[ri] = new_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    new_diffs = _diffs_set(new_row)
                    all_used |= new_diffs
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    static_diffs[ri] = _build_static_diffs(new_row, k)
                    row_maxes[ri] = new_row_max
                    current_val = new_global_max
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # --- Local SA Move ---
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(20, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
                
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
            max_increase = min(10, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
                
            if random.random() < 0.1:
                low = min_allowed
                high = max_allowed
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
                    static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
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
        
        # Restart if stuck
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = set()
            for row in current_rows:
                all_used |= _diffs_set(row)
            row_diffs = [_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 50.0
            no_improve = 0

    # 3. Final Polish Phase: Aggressive Tightening
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = set()
    for row in current_rows:
        all_used |= _diffs_set(row)
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    polish_deadline = min(deadline, start_time + 148.0)
    
    for _ in range(10):
        if time.time() > polish_deadline:
            break
        improved = False
        for ri in range(n_rows):
            forbidden = set()
            for j in range(n_rows):
                if j != ri:
                    forbidden |= row_diffs[j]
            
            target_max = row_maxes[ri] - 1
            if target_max < 0:
                target_max = best_val
            
            # Try backtracking for better quality
            new_row = _backtracking_row(k, forbidden, target_max, max_time=0.2)
            
            if new_row is None:
                new_row = _randomized_greedy_row(k, forbidden, target_max, max_candidates=20)
            
            if new_row is not None:
                new_row_max = max(new_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = new_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                if new_global_max < current_val:
                    all_used -= row_diffs[ri]
                    new_diffs = _diffs_set(new_row)
                    all_used |= new_diffs
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    static_diffs[ri] = _build_static_diffs(new_row, k)
                    row_maxes[ri] = new_row_max
                    current_val = new_global_max
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    improved = True
                    break
            else:
                new_row = _randomized_greedy_row(k, forbidden, best_val, max_candidates=10)
                if new_row is not None and max(new_row) < row_maxes[ri]:
                    new_row_max = max(new_row)
                    old_row_max = row_maxes[ri]
                    row_maxes[ri] = new_row_max
                    new_global_max = max(row_maxes)
                    row_maxes[ri] = old_row_max
                    
                    if new_global_max <= current_val:
                        all_used -= row_diffs[ri]
                        new_diffs = _diffs_set(new_row)
                        all_used |= new_diffs
                        current_rows[ri] = new_row
                        row_diffs[ri] = new_diffs
                        static_diffs[ri] = _build_static_diffs(new_row, k)
                        row_maxes[ri] = new_row_max
                        current_val = new_global_max
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        improved = True
                        break
        if not improved:
            break

    # Final local search squeeze
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = set()
    for row in current_rows:
        all_used |= _diffs_set(row)
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    T = 1.0
    T_min = 0.1
    cool = 0.99
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < polish_deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(5, old_val - min_allowed)
        
        candidates = [old_val - d for d in range(1, max_decrease + 1)]
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
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.05:
                    accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
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

    return best_rows

# EVOLVE-BLOCK-END
