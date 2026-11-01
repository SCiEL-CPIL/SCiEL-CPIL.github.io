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
    This allows O(k) check for new differences when moving position pi.
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

def _check_move_validity(row, pi, new_val, static_diffs_pi, all_used, old_diffs):
    """
    Check if placing new_val at position pi in row is valid.
    1. Internal validity: new diffs involving pi don't collide with static diffs of the row.
    2. External validity: new diffs involving pi don't collide with all_used (excluding old_diffs of this row).
    
    Returns (new_diffs_set, is_valid).
    """
    n = len(row)
    # Start with static diffs (diffs not involving pi)
    new_diffs = set(static_diffs_pi)
    
    # Check diffs involving pi
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
            
        # Internal collision check
        if d in new_diffs:
            return None, False
        
        # External collision check: 
        # d must not be in all_used unless it was in old_diffs (which is being replaced)
        # Note: We are replacing old_diffs with new_diffs. 
        # So d is allowed if it's in old_diffs (it was there before) OR if it's not in all_used at all.
        # But wait, if d is in old_diffs, it's already in all_used. 
        # The condition is: (all_used - old_diffs) must be disjoint from new_diffs.
        # Equivalently: new_diffs & (all_used - old_diffs) == empty set.
        if d in all_used and d not in old_diffs:
            return None, False
            
        new_diffs.add(d)
        
    return new_diffs, True

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing Phase 1: General Search
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = 300.0
    T_min = 1.0
    cool = 0.9999
    
    no_improve = 0
    max_no_improve = 20000
    
    # Precompute row diffs and static diffs
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        
        # Sample a subset to speed up, but ensure we try the max decrease
        if max_decrease > 10:
            n_try = min(max_decrease, random.randint(10, 20))
            try_decreases = random.sample(range(1, max_decrease + 1), n_try)
            if max_decrease not in try_decreases:
                try_decreases.append(max_decrease)
            for d in try_decreases:
                candidates.append(old_val - d)
        else:
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
                
        # Increasing candidates (for exploration)
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
        max_increase = min(20, max_allowed - old_val)
        if max_increase > 0:
            n_try_inc = min(max_increase, random.randint(5, 10))
            try_increases = random.sample(range(1, max_increase + 1), n_try_inc)
            for d in try_increases:
                candidates.append(old_val + d)
        
        # Random jump
        if random.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
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
            # Basic validity check for this position (strictly increasing)
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast diff computation and validity check
            new_diffs, valid = _check_move_validity(
                current_rows[ri], pi, new_val, static_pi, all_used, old_diffs)
            
            if not valid:
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
                if random.random() < 0.3:  # Allow some neutral moves
                    accept = True
            else:
                # Accept worse moves with probability
                if T > T_min:
                    prob = math.exp(-delta / max(T, 1.0))
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
                
                # Rebuild static diffs for this row (expensive but necessary for correctness of future checks)
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
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 300.0
            no_improve = 0

    # 3. Targeted Shrink Phase: Focus on reducing the maximum
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
    max_no_improve_polish = 50000
    
    while time.time() < deadline and no_improve < max_no_improve_polish:
        # Bias toward rows with the maximum entry
        if max_row_indices and random.random() < 0.9:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        # For the max row, prioritize the last position (which is the max)
        if ri in max_row_indices and random.random() < 0.7:
            pi = k  # the last element
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(100, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try a random subset, biased toward larger decreases
        if max_decrease > 20:
            n_try = min(max_decrease, random.randint(15, 30))
            # Weight toward larger decreases by sampling from the upper half more often
            try_decreases = []
            for _ in range(n_try):
                if random.random() < 0.5:
                    try_decreases.append(random.randint(max_decrease//2, max_decrease))
                else:
                    try_decreases.append(random.randint(1, max_decrease))
            # Ensure we try the largest
            if max_decrease not in try_decreases:
                try_decreases.append(max_decrease)
        else:
            try_decreases = list(range(1, max_decrease + 1))
        
        candidates = [old_val - d for d in try_decreases]
        random.shuffle(candidates)
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid = _check_move_validity(
                current_rows[ri], pi, new_val, static_pi, all_used, old_diffs)
            
            if not valid:
                continue
                
            cand_row_max = new_val if pi == k else current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            # In polish phase, only accept improving or (rarely) neutral moves
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
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1

    return best_rows

# EVOLVE-BLOCK-END
