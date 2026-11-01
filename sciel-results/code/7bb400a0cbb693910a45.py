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
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _validate_move(cand_row, old_diffs, all_used):
    """
    Check if changing a row to cand_row is valid (no internal collisions, no cross-row collisions).
    Returns (new_diffs, is_valid).
    """
    n = len(cand_row)
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

def _try_greedy_row(k, all_used, max_limit):
    """
    Attempt to construct a single valid row of length k+1 starting at 0,
    using only differences not in all_used, and keeping max entry <= max_limit.
    Returns the row if successful, None otherwise.
    """
    row = [0]
    x = 0
    used_in_row = set()
    while len(row) < k + 1:
        x += 1
        if x > max_limit:
            return None
        new_diffs = {x - a for a in row}
        # Check internal conflicts
        if new_diffs & used_in_row:
            continue
        # Check global conflicts
        if new_diffs & all_used:
            continue
        row.append(x)
        used_in_row |= new_diffs
    return row

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to return by 145s.
    deadline = start_time + 145.0
    
    # 1. Initialization
    # Run a few greedy inits to find a good starting point
    best_rows = None
    best_val = float('inf')
    
    for _ in range(10):
        rows = _greedy_init(n_rows, k)
        val = _global_max(rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
            
    # If greedy is too slow or fails, fallback to random (unlikely for small k)
    if best_rows is None:
        # Random fallback
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Random search for next element
                for _attempt in range(1000):
                    x = random.randint(1, 200)
                    if x <= row[-1]:
                        continue
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        continue
                    row.append(x)
                    used |= new_diffs
                    break
            best_rows.append(row)
        best_val = _global_max(best_rows)

    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    row_maxes = [max(r) for r in current_rows]
    all_used = _all_used(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    
    # 2. Local Search Phase
    # We use a combination of:
    # a) Single element tweaks (Simulated Annealing style)
    # b) Two-element tweaks (more powerful)
    # c) Row replacement (greedy reconstruction of one row)
    
    T = 100.0
    T_min = 0.5
    cool = 0.9999
    no_improve = 0
    max_no_improve = 20000
    
    # Track best
    global_best_val = best_val
    global_best_rows = [r[:] for r in best_rows]
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from best with perturbation
            current_rows = [r[:] for r in global_best_rows]
            current_val = global_best_val
            all_used = _all_used(current_rows)
            row_diffs = [_row_diffs(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 50.0
            no_improve = 0
            
            # Perturb: pick a random row and change a few elements
            for _ in range(3):
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                # Try to find a valid new value for this position
                old_val = current_rows[ri][pi]
                # Just shift it slightly
                new_val = old_val + random.randint(-5, 5)
                if new_val > current_rows[ri][pi-1] and (pi == k or new_val < current_rows[ri][pi+1]):
                    current_rows[ri][pi] = new_val
                    # Re-validate and update structures if valid, else revert
                    rd = _row_diffs(current_rows[ri])
                    # Simple check: if internal conflict, revert
                    if len(rd) != len(current_rows[ri])*(len(current_rows[ri])-1)//2:
                         current_rows[ri][pi] = old_val
                    else:
                         # Check global conflict
                         if rd & all_used:
                             # Remove old diffs, add new, check if any overlap with other rows
                             # This is expensive, so just revert if overlap
                             if rd & (all_used - row_diffs[ri]):
                                  current_rows[ri][pi] = old_val
                             else:
                                  all_used -= row_diffs[ri]
                                  all_used |= rd
                                  row_diffs[ri] = rd
                                  row_maxes[ri] = max(current_rows[ri])
                                  current_val = max(row_maxes)
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.6:
            # Single element move
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            # Generate candidates
            candidates = []
            # Try values close to current
            for d in range(1, 20):
                candidates.append(old_val - d)
                candidates.append(old_val + d)
            # Random jump
            if random.random() < 0.2:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                new_diffs, valid = _validate_move(cand_row, row_diffs[ri], all_used)
                if not valid:
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
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    
                    if current_val < global_best_val:
                        global_best_val = current_val
                        global_best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
                T = max(T_min, T * cool)
                
        elif move_type < 0.9:
            # Two-element move in the same row
            ri = random.randrange(n_rows)
            # Pick two distinct indices
            p1, p2 = random.sample(range(1, k + 1), 2)
            if p1 > p2:
                p1, p2 = p2, p1
            
            old_val1 = current_rows[ri][p1]
            old_val2 = current_rows[ri][p2]
            
            # Generate candidates for both
            cands1 = [old_val1 + d for d in range(-10, 11) if d != 0]
            cands2 = [old_val2 + d for d in range(-10, 11) if d != 0]
            
            # Filter valid boundaries
            cands1 = [v for v in cands1 if v > current_rows[ri][p1-1] and (p1 == k or v < current_rows[ri][p1+1])]
            cands2 = [v for v in cands2 if v > current_rows[ri][p2-1] and (p2 == k or v < current_rows[ri][p2+1])]
            
            if not cands1 or not cands2:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            # Try a few random pairs
            moved = False
            for _ in range(5):
                nv1 = random.choice(cands1)
                nv2 = random.choice(cands2)
                
                cand_row = current_rows[ri][:]
                cand_row[p1] = nv1
                cand_row[p2] = nv2
                
                # Check strict increase
                valid_order = True
                for i in range(len(cand_row)-1):
                    if cand_row[i] >= cand_row[i+1]:
                        valid_order = False
                        break
                if not valid_order:
                    continue
                
                new_diffs, valid = _validate_move(cand_row, row_diffs[ri], all_used)
                if not valid:
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
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    
                    if current_val < global_best_val:
                        global_best_val = current_val
                        global_best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
                T = max(T_min, T * cool)
                
        else:
            # Row Replacement: Try to replace one row with a greedy one
            ri = random.randrange(n_rows)
            # Temporarily remove this row's diffs from all_used
            old_diffs = row_diffs[ri]
            temp_all_used = all_used - old_diffs
            
            # Try to build a new row
            new_row = _try_greedy_row(k, temp_all_used, global_best_val)
            
            if new_row:
                # Check if it's actually better or equal
                new_row_max = max(new_row)
                old_row_max = row_maxes[ri]
                
                # Only accept if it reduces the global max or is same max but smaller row max
                # Or just accept if it's valid and doesn't increase global max too much
                if new_row_max < old_row_max:
                    # It's a clear win for this row
                    all_used = temp_all_used | _row_diffs(new_row)
                    current_rows[ri] = new_row
                    row_diffs[ri] = _row_diffs(new_row)
                    row_maxes[ri] = new_row_max
                    current_val = max(row_maxes)
                    
                    if current_val < global_best_val:
                        global_best_val = current_val
                        global_best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                elif new_row_max == old_row_max:
                    # Same max, maybe better distribution?
                    if random.random() < 0.1:
                        all_used = temp_all_used | _row_diffs(new_row)
                        current_rows[ri] = new_row
                        row_diffs[ri] = _row_diffs(new_row)
                        row_maxes[ri] = new_row_max
                        current_val = max(row_maxes)
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
    return global_best_rows

# EVOLVE-BLOCK-END
