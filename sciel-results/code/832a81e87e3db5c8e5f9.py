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

def _reconstruct_row(k, forbidden_diffs, max_val_limit=None, randomize_prob=0.0):
    """
    Reconstruct a row of k+1 elements starting at 0, avoiding forbidden_diffs.
    Returns the row if valid, else None.
    """
    row = [0]
    internal_diffs = set()
    
    for _ in range(k):
        found = False
        x = row[-1] + 1
        
        # Determine search limit
        if max_val_limit is not None:
            search_limit = max_val_limit
        else:
            search_limit = 200 + max(row)
            
        # Collect valid candidates up to a small window to allow randomization
        valid_candidates = []
        
        while x <= search_limit:
            new_diffs = {x - a for a in row}
            
            # Check internal uniqueness
            if len(new_diffs) != len(row):
                x += 1
                continue
                
            # Check against forbidden
            if new_diffs & forbidden_diffs:
                x += 1
                continue
                
            valid_candidates.append(x)
            
            # If we are greedy, take the first one.
            # If randomize_prob > 0, collect a few.
            if randomize_prob == 0.0:
                row.append(x)
                internal_diffs |= new_diffs
                found = True
                break
            else:
                # Collect up to 5 candidates
                if len(valid_candidates) >= 5:
                    break
            x += 1
        
        if not found:
            if randomize_prob > 0.0 and valid_candidates:
                # Pick a random candidate
                x = random.choice(valid_candidates)
                new_diffs = {x - a for a in row}
                row.append(x)
                internal_diffs |= new_diffs
                found = True
            else:
                return None
                
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    row_diffs = [_row_diffs_set(r) for r in rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    best_rows = [r[:] for r in rows]
    best_val = max(max(r) for r in rows)
    
    # 2. Iterated Row Reconstruction
    T = 100.0
    T_min = 0.1
    cool = 0.999
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        
        current_diffs = row_diffs[ri]
        forbidden = all_diffs - current_diffs
        
        # Try to reconstruct with a limit slightly better than current
        # If we fail, try without limit
        current_max = max(rows[ri])
        
        # Try to find a row with max < current_max
        new_row = _reconstruct_row(k, forbidden, max_val_limit=current_max - 1, randomize_prob=0.2)
        
        if new_row is None:
            # Try to find a row with max <= current_max (to shuffle)
            new_row = _reconstruct_row(k, forbidden, max_val_limit=current_max, randomize_prob=0.5)
            
        if new_row:
            new_diffs = _row_diffs_set(new_row)
            new_max = max(new_row)
            
            delta = new_max - current_max
            
            accept = False
            if new_max < current_max:
                accept = True
            elif new_max == current_max:
                if new_row != rows[ri] and random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_diffs -= current_diffs
                all_diffs |= new_diffs
                rows[ri] = new_row
                row_diffs[ri] = new_diffs
                
                current_global_max = max(max(r) for r in rows)
                
                if current_global_max < best_val:
                    best_val = current_global_max
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > 1000:
            rows = [r[:] for r in best_rows]
            row_diffs = [_row_diffs_set(r) for r in rows]
            all_diffs = set()
            for rd in row_diffs:
                all_diffs |= rd
            T = 100.0
            no_improve = 0

    return best_rows

# EVOLVE-BLOCK-END
