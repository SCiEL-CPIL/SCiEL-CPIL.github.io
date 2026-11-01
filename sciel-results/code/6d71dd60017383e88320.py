import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Compute set of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _global_max(rows):
    return max(max(row) for row in rows)

def _validate_rows(rows, k):
    """Check all rows are valid (strictly increasing, start at 0) and all diffs distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        n = len(row)
        row_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False
                if d in used:
                    return False
                row_diffs.add(d)
        used |= row_diffs
    return True

def _random_construct(n_rows, k, max_entry, seed=None):
    """
    Randomized construction: build rows one by one using random sampling
    with backtracking-like pruning.
    """
    if seed is not None:
        random.seed(seed)
    
    rows = []
    used = set()
    
    for r in range(n_rows):
        row = None
        for attempt in range(200):
            row = _random_row(k, max_entry, used)
            if row is not None:
                break
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs_set(row)
    
    return rows

def _random_row(k, max_entry, used):
    """Build a single random row of k+1 increasing integers starting at 0."""
    row = [0]
    current_diffs = set()
    for pos in range(1, k + 1):
        # Determine range for next value
        min_val = row[-1] + 1
        remaining = k + 1 - pos
        max_val = max_entry - (remaining - 1)
        
        if min_val > max_val:
            return None
        
        # Try random values in range, prefer smaller ones
        found = False
        # Try a few random positions
        for _ in range(50):
            if min_val == max_val:
                v = min_val
            else:
                # Bias toward smaller values
                t = random.random() ** 1.5
                v = int(min_val + t * (max_val - min_val))
                v = max(min_val, min(max_val, v))
            
            new_diffs = {v - a for a in row}
            if len(new_diffs) != len(row):
                continue
            if new_diffs & current_diffs:
                continue
            if new_diffs & used:
                continue
            
            row.append(v)
            current_diffs |= new_diffs
            found = True
            break
        
        if not found:
            return None
    
    return row

def _greedy_init(n_rows, k):
    """
    Greedy initialization with some randomization for diversity.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, time_limit, max_scope=None):
    """
    Local search with multi-point moves.
    Objective: minimize global max, with penalty for conflicts.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    # Precompute
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    all_used = set()
    for rd in row_diffs:
        all_used |= rd
    row_maxes = [max(r) for r in current_rows]
    
    T = 5.0
    T_min = 0.01
    cool = 0.99995
    no_improve = 0
    
    while time.time() < deadline:
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Single point move
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else max_scope if max_scope else current_val + 20
            
            if min_allowed > max_allowed:
                continue
            
            # Try random new value
            if random.random() < 0.6:
                # Try to decrease
                new_val = random.randint(min_allowed, min(old_val - 1, max_allowed)) if old_val > min_allowed else old_val
            else:
                new_val = random.randint(min_allowed, max_allowed)
            
            if new_val == old_val:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Validate
            new_diffs, valid = _validate_row_move(cand_row, row_diffs[ri], all_used)
            if not valid:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            cand_row_max = max(cand_row)
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
            else:
                prob = math.exp(-delta / max(T, 0.01))
                if random.random() < prob:
                    accept = True
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.8:
            # Two-point move in same row
            ri = random.randrange(n_rows)
            p1, p2 = random.sample(range(1, k + 1), 2)
            if p2 < p1:
                p1, p2 = p2, p1
            
            old_v1 = current_rows[ri][p1]
            old_v2 = current_rows[ri][p2]
            
            min_v1 = current_rows[ri][p1 - 1] + 1
            max_v1 = current_rows[ri][p1 + 1] - 1 if p1 < k else (max_scope if max_scope else current_val + 20)
            min_v2 = current_rows[ri][p2 - 1] + 1
            max_v2 = (max_scope if max_scope else current_val + 20) if p2 == k else current_rows[ri][p2 + 1] - 1
            
            # Ensure ordering
            if p2 == p1 + 1:
                max_v1 = min(max_v1, old_v2 - 1)
                min_v2 = max(min_v2, old_v1 + 1)
            
            if min_v1 > max_v1 or min_v2 > max_v2:
                continue
            
            new_v1 = random.randint(min_v1, max_v1)
            new_v2 = random.randint(min_v2, max_v2)
            
            if p2 == p1 + 1 and new_v1 >= new_v2:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[p1] = new_v1
            cand_row[p2] = new_v2
            
            # Check strict increase
            valid_incr = True
            for i in range(1, len(cand_row)):
                if cand_row[i] <= cand_row[i-1]:
                    valid_incr = False
                    break
            if not valid_incr:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            new_diffs, valid = _validate_row_move(cand_row, row_diffs[ri], all_used)
            if not valid:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            cand_row_max = max(cand_row)
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
            else:
                prob = math.exp(-delta / max(T, 0.01))
                if random.random() < prob:
                    accept = True
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Row swap: try to replace an entire row with a new random one
            ri = random.randrange(n_rows)
            max_scope = best_val
            new_row = _random_row(k, max_scope, all_used - row_diffs[ri])
            if new_row is None:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            new_diffs = _row_diffs_set(new_row)
            # Check no conflicts with other rows
            if new_diffs & (all_used - row_diffs[ri]):
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            cand_row_max = max(new_row)
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
            else:
                prob = math.exp(-delta / max(T, 0.01))
                if random.random() < prob:
                    accept = True
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = new_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        T = max(T_min, T * cool)
        
        if no_improve > 20000:
            break
    
    return best_rows

def _validate_row_move(cand_row, old_diffs, all_used):
    """
    Check if cand_row is valid (strictly increasing, no internal diff conflicts, no cross-row conflicts).
    Returns (new_diffs, is_valid).
    """
    n = len(cand_row)
    # Check strictly increasing
    for i in range(1, n):
        if cand_row[i] <= cand_row[i-1]:
            return None, False
    
    new_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = cand_row[j] - cand_row[i]
            if d in new_diffs:
                return None, False
            new_diffs.add(d)
    
    # Check cross-row conflicts
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return None, False
    
    return new_diffs, True

def construct(n_rows, k):
    start_time = time.time()
    total_deadline = start_time + 140.0  # Safety margin
    
    # 1. Greedy initialization
    best_rows = _greedy_init(n_rows, k)
    best_val = _global_max(best_rows)
    
    # 2. Iterated local search with multiple random starts
    n_restarts = 0
    while time.time() < total_deadline - 10.0:
        # Create a perturbed start
        if n_restarts == 0:
            start_rows = [r[:] for r in best_rows]
        else:
            # Random construction with scope constraint
            max_scope = best_val
            start_rows = _random_construct(n_rows, k, max_scope, seed=int(time.time()*1000) + n_restarts)
            if start_rows is None:
                # Fallback: perturb best
                start_rows = [r[:] for r in best_rows]
                for _ in range(n_rows * k):
                    ri = random.randrange(n_rows)
                    pi = random.randrange(1, k + 1)
                    min_v = start_rows[ri][pi-1] + 1
                    max_v = (start_rows[ri][pi+1] - 1) if pi < k else best_val + 5
                    if min_v <= max_v:
                        start_rows[ri][pi] = random.randint(min_v, max_v)
                # Ensure validity by fixing
                if not _validate_rows(start_rows, k):
                    start_rows = [r[:] for r in best_rows]
        
        # Local search
        ls_time = min(30.0, total_deadline - time.time() - 5.0)
        if ls_time < 5.0:
            break
        
        result = _local_search(start_rows, n_rows, k, ls_time, max_scope=best_val)
        
        if result is not None:
            val = _global_max(result)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in result]
        
        n_restarts += 1
    
    # Final validation
    if not _validate_rows(best_rows, k):
        # Fallback to greedy
        best_rows = _greedy_init(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
