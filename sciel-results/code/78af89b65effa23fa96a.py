import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init_randomized(n_rows, k, rand_width=5):
    """
    Randomized greedy initialization.
    For each row, pick the smallest valid next number, but with some randomness
    from the smallest 'rand_width' valid options to diversify the solution space.
    """
    rows = []
    used = set()
    
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Find valid candidates in a window
            valid_cands = []
            check_x = x
            # We can just iterate from x upwards until we find 'rand_width' valid ones
            # But x is the smallest possible. Let's just check x, x+1, ...
            # Optimization: Just check x first. If valid, add to list.
            # To keep it simple and fast, let's just do standard greedy but 
            # occasionally jump. 
            # Actually, let's stick to standard greedy for speed in init, 
            # but run it multiple times with different random seeds for row order?
            # No, row order doesn't matter for the final set, only for the 'used' set evolution.
            # The 'used' set depends on the order.
            
            # Let's try: Pick x. Check if valid.
            # If valid, pick it.
            # To randomize, we can sometimes pick x+d for small d if valid.
            
            found = False
            # Try a small window of candidates starting from x
            # But x is the smallest integer greater than previous.
            # So candidates are x, x+1, ..., x+rand_width
            for delta in range(rand_width + 1):
                cand = x + delta
                new_diffs = {cand - a for a in row}
                if not (new_diffs & used):
                    # Valid
                    if random.random() < 0.8 or delta == 0:
                        # Accept this candidate
                        row.append(cand)
                        used |= new_diffs
                        x = cand # Next search starts after this
                        found = True
                        break
                    else:
                        # Skip this one, try next? 
                        # This logic is flawed because if we skip x, x+1 might be valid.
                        # Let's just collect valids in a window and pick one.
                        pass
            
            if not found:
                 # Fallback: standard greedy from x upwards
                 while True:
                     x += 1
                     new_diffs = {x - a for a in row}
                     if not (new_diffs & used):
                         row.append(x)
                         used |= new_diffs
                         break

        rows.append(row)
        
    return rows

def _compute_new_diffs_fast(row, pi, new_val):
    """
    Compute the set of differences involving index pi if row[pi] was new_val.
    Returns the set of diffs.
    """
    diffs = set()
    for i in range(len(row)):
        if i == pi:
            continue
        diffs.add(abs(new_val - row[i]))
    return diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization: Run randomized greedy a few times
    best_rows = None
    best_val = float('inf')
    
    # Try a few initializations
    for _ in range(10):
        rows = _greedy_init_randomized(n_rows, k, rand_width=3)
        val = max(max(r) for r in rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
            
        if time.time() > start_time + 5.0:
            break

    # Prepare state for SA
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = set()
    row_diffs = []
    for r in current_rows:
        d = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d.add(r[j] - r[i])
        row_diffs.append(d)
        all_used |= d
        
    row_maxes = [max(r) for r in current_rows]
    
    # 2. Simulated Annealing
    T = 100.0
    T_min = 0.1
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Select row
        # Bias towards rows with large max
        if random.random() < 0.5:
            # Pick the row with the global max (or one of them)
            max_val = max(row_maxes)
            candidates = [i for i in range(n_rows) if row_maxes[i] == max_val]
            ri = random.choice(candidates)
        else:
            ri = random.randrange(n_rows)
            
        # Select position
        # Bias towards the end of the row
        if random.random() < 0.5:
            pi = k # Last element
        else:
            pi = random.randrange(1, k + 1)
            
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        # We want to reduce the max, so try smaller values first.
        # But also allow some exploration.
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        
        candidates = []
        
        # Decreasing moves
        if old_val > min_allowed:
            # Try a few random decreasing steps
            max_dec = min(20, old_val - min_allowed)
            for _ in range(5):
                d = random.randint(1, max_dec)
                candidates.append(old_val - d)
            # Always try old_val - 1
            candidates.append(old_val - 1)
            
        # Increasing moves (for exploration)
        if old_val < max_allowed:
            max_inc = min(10, max_allowed - old_val)
            if max_inc > 0:
                for _ in range(2):
                    d = random.randint(1, max_inc)
                    candidates.append(old_val + d)
                    
        # Random jump
        if random.random() < 0.1:
            candidates.append(random.randint(min_allowed, max_allowed))
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast check
            old_diffs_pi = _compute_new_diffs_fast(current_rows[ri], pi, old_val)
            new_diffs_pi = _compute_new_diffs_fast(current_rows[ri], pi, new_val)
            
            # Internal validity: new_diffs_pi must have size k
            if len(new_diffs_pi) != k:
                continue
                
            # External validity: new_diffs_pi must not intersect (all_used - old_diffs_pi)
            # all_used contains old_diffs_pi.
            # So we check if any d in new_diffs_pi is in all_used AND NOT in old_diffs_pi.
            conflict = False
            for d in new_diffs_pi:
                if d in all_used and d not in old_diffs_pi:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Move is valid
            # Compute new global max
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
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
                accept = random.random() < 0.2
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs_pi
                all_used |= new_diffs_pi
                current_rows[ri] = cand_row
                row_diffs[ri] = _compute_new_diffs_fast(cand_row, pi, new_val) # This is only diffs involving pi? 
                # Wait, row_diffs[ri] should be the FULL set of diffs for the row for future checks?
                # In my fast check, I only computed diffs involving pi.
                # The check `d not in old_diffs_pi` works because `old_diffs_pi` are the ones being removed.
                # The rest of the diffs in the row are in `all_used` and are NOT being removed, 
                # so if `new_diffs_pi` intersects with them, `d in all_used` is True and `d not in old_diffs_pi` is True -> Conflict. Correct.
                
                # So we don't need to store the full row_diffs set for the check, 
                # we just need the diffs involving the modified index.
                # However, if we modify a different index later, we need the diffs involving THAT index.
                # The fast check computes `_compute_new_diffs_fast` on the fly.
                # So we don't need to maintain `row_diffs` at all! We just compute it when needed.
                
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
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
        if no_improve > 20000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = set()
            for r in current_rows:
                d = set()
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d.add(r[j] - r[i])
                all_used |= d
            row_maxes = [max(r) for r in current_rows]
            T = 100.0
            no_improve = 0

    return best_rows

# EVOLVE-BLOCK-END
