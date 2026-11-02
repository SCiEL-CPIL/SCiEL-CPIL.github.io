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

def _is_move_valid(row, pi, new_val, static_pi, old_diffs, all_used):
    """
    Fast validity check without creating the new_diffs set.
    Returns True if the move is valid.
    """
    n = len(row)
    new_diffs_list = []
    
    # Compute new diffs involving pi
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        new_diffs_list.append(d)
        
    # Check internal validity
    seen_new = set()
    for d in new_diffs_list:
        # Must not be in static diffs (which are unchanged)
        if d in static_pi:
            return False
        # Must not be duplicate among new diffs
        if d in seen_new:
            return False
        seen_new.add(d)
        
    # Check global conflicts
    # A conflict exists if a new diff is in all_used but was NOT in old_diffs
    for d in new_diffs_list:
        if d in all_used and d not in old_diffs:
            return False
            
    return True

def _get_new_diffs_set(row, pi, new_val, static_pi):
    """
    Construct the new_diffs set. Only called if move is accepted.
    """
    n = len(row)
    new_diffs = set(static_pi)
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        new_diffs.add(d)
    return new_diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    best_sum_max = sum(max(r) for r in best_rows)
    
    # State
    current_rows = [r[:] for r in rows]
    current_val = best_val
    current_sum_max = best_sum_max
    
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    # SA Parameters
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    no_improve = 0
    max_no_improve = 30000
    
    # Phase 1: Main SA loop
    sa_deadline = deadline - 18.0  # Reserve 18s for polish phases
    
    while time.time() < sa_deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
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
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast validity check (no set construction)
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            # Compute new metrics
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
                
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            row_maxes[ri] = old_row_max
            
            # Acceptance with secondary objective
            accept = False
            if new_global_max < current_val:
                accept = True
            elif new_global_max == current_val:
                if new_sum_max < current_sum_max:
                    accept = True
                elif new_sum_max == current_sum_max:
                    accept = random.random() < 0.1
                else:
                    accept = random.random() < 0.01
            else:
                delta = new_global_max - current_val
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    accept = random.random() < prob
            
            if accept:
                # Build the diff set only on acceptance
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
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
            current_sum_max = best_sum_max
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # Phase 2: Targeted shrink phase - focus on rows with maximum entry
    polish_deadline = min(deadline, start_time + 113.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    current_sum_max = best_sum_max
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
            pi = k
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try from largest decrease to smallest
        candidates = [old_val - d for d in range(max_decrease, 0, -1)]
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            # In polish phase, only accept improving or neutral moves
            if delta < 0:
                accept = True
            elif delta == 0:
                if new_sum_max < current_sum_max:
                    accept = True
                else:
                    accept = random.random() < 0.1
            else:
                accept = False
            
            if accept:
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
    
    # Phase 3: Final quick SA pass with very low temperature
    final_deadline = deadline
    T = 0.2
    T_min = 0.01
    cool = 0.99999
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    current_sum_max = best_sum_max
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
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.05
            else:
                accept = False
            
            if accept:
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
                    best_rows = [r[:] for r in current_rows]
                    
                moved = True
                break
    
    return best_rows

# EVOLVE-BLOCK-END
