import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init_randomized(n_rows, k, rand_width=5):
    """
    Randomized greedy initialization.
    """
    rows = []
    used = set()
    
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            found = False
            # Try a small window of candidates starting from x
            for delta in range(rand_width + 1):
                cand = x + delta
                new_diffs = {cand - a for a in row}
                if not (new_diffs & used):
                    # Accept this candidate
                    row.append(cand)
                    used |= new_diffs
                    x = cand
                    found = True
                    break
            
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

def _rebuild_row(row_idx, n_rows, k, all_used, current_rows):
    """
    Attempts to rebuild a single row to minimize its max value,
    given the constraints from all other rows.
    Returns the new row if successful and better, else None.
    """
    # Calculate used diffs excluding the current row
    # To do this efficiently, we need to know the diffs of the current row.
    old_row = current_rows[row_idx]
    old_diffs = set()
    for i in range(len(old_row)):
        for j in range(i+1, len(old_row)):
            old_diffs.add(old_row[j] - old_row[i])
    
    # Remove old row diffs from all_used to get 'external_used'
    # Note: all_used is a set. We can't easily subtract a set of diffs 
    # if we don't have the set of diffs precomputed or if we want to avoid O(N) copy.
    # But for k=6, N is small. 
    # Better: Pass 'external_used' as a set computed on the fly? 
    # Or, modify construct to maintain 'used' sets per row?
    
    # For this helper, let's assume we can compute the "forbidden" set.
    # The forbidden set for the new row is (all_used - old_diffs).
    forbidden = all_used - old_diffs
    
    # Greedy construction of the new row
    new_row = [0]
    x = 0
    success = True
    while len(new_row) < k + 1:
        x += 1
        # Limit search space: don't go beyond the current max of this row + some margin
        # If we can't find a valid x within a reasonable bound, fail.
        if x > max(old_row) + 10:
            success = False
            break
            
        new_diffs = {x - a for a in new_row}
        if not (new_diffs & forbidden):
            new_row.append(x)
            # Note: we don't update 'forbidden' here because we are building 
            # the row in isolation against the *other* rows' constraints.
            # The internal diffs of new_row must be distinct, which is guaranteed 
            # by the "new_row is a Sidon sequence" property we are trying to build?
            # Wait. The requirement is that ALL differences in the row are distinct 
            # from each other AND from other rows.
            # My greedy check `new_diffs & forbidden` ensures no conflict with OTHER rows.
            # Does it ensure internal distinctness?
            # new_diffs contains {x - a for a in new_row}.
            # If new_row is built greedily such that all previous differences were distinct,
            # and we check against 'forbidden' (which doesn't include new_row's own diffs yet),
            # we might pick an x such that x-a1 == x-a2? No, a1!=a2 implies x-a1 != x-a2.
            # We might pick x such that x-a1 == b-j? That's covered by forbidden.
            # We might pick x such that x-a1 == c-d where c,d are in new_row?
            # If c,d are in new_row, then c-d is an internal diff.
            # Is it in 'forbidden'? No, forbidden is from OTHER rows.
            # So we MUST ensure internal distinctness.
            
            # Let's re-verify:
            # We need new_row to be a Sidon sequence (distinct internal diffs).
            # The greedy step: pick smallest x > prev such that {x-a | a in new_row} are distinct 
            # AND disjoint from 'forbidden'.
            # The set {x-a | a in new_row} are automatically distinct from each other 
            # because a's are distinct.
            # So we just need to check disjoint from 'forbidden'.
            # BUT, we also need to ensure that the NEW diffs don't collide with 
            # EXISTING internal diffs of new_row.
            # Existing internal diffs are not in 'forbidden'.
            # So we must check: new_diffs & (internal_diffs_of_new_row_so_far) == empty?
            # Actually, if we build greedily, do we guarantee internal distinctness?
            # Example: new_row = [0, 3, 5]. Diffs: 3, 5, 2.
            # Next x. If x=7. Diffs: 7, 4, 2. 
            # 2 is already an internal diff (5-3=2).
            # So {x-a} contains 2. 2 is in internal_diffs.
            # My check `new_diffs & forbidden` would PASS if 2 is not in forbidden.
            # But 2 is an internal diff! So the row would have duplicate diff 2.
            # Thus, we must also check against the set of internal diffs of the current partial row.
            
            # Let's optimize: maintain `internal_used` for the row being built.
            # If `new_diffs & (forbidden | internal_used)` is empty, accept.
            
            # I will rewrite the loop inside _rebuild_row to be correct.
            pass # Placeholder, will replace below

    if not success:
        return None
        
    # Verify full validity of new_row against forbidden and internal
    # And check if max is better
    if max(new_row) < max(old_row):
        return new_row
    return None

def _compute_row_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _rebuild_row_correct(row_idx, n_rows, k, all_used, current_rows):
    """
    Correctly rebuilds a row to minimize max, respecting internal and external constraints.
    """
    old_row = current_rows[row_idx]
    old_diffs = _compute_row_diffs(old_row)
    forbidden = all_used - old_diffs
    
    new_row = [0]
    internal_used = set()
    x = 0
    max_allowed = max(old_row) # Try to stay within or below current max
    
    while len(new_row) < k + 1:
        found = False
        # Search from x+1 upwards
        # To speed up, we can limit search to max_allowed + small buffer
        # If we exceed max_allowed, we can still try a bit more to see if a better
        # arrangement exists that pushes max up slightly but allows completion?
        # No, we want to MINIMIZE max. So if we can't complete within max_allowed,
        # we should fail (return None) to keep the old row, OR we can try to complete
        # with a higher max if the old row was already bad?
        # The goal is to improve. So if we can't do <= old_max, we should probably
        # just return None unless we are desperate.
        # Let's allow search up to old_max.
        
        for cand in range(x + 1, max_allowed + 1):
            new_diffs = {cand - a for a in new_row}
            # Check internal distinctness
            if new_diffs & internal_used:
                continue
            # Check external distinctness
            if new_diffs & forbidden:
                continue
            
            # Valid
            new_row.append(cand)
            internal_used |= new_diffs
            x = cand
            found = True
            break
            
        if not found:
            return None
            
    # If we reached here, new_row is valid and max(new_row) <= max(old_row)
    # We only accept if strictly better or equal (but we want to minimize, so strictly better is ideal)
    # Actually, if max is equal, it doesn't help the global score.
    # But it might free up differences that allow OTHER rows to shrink?
    # Yes, so we should accept if max is <= old_max.
    return new_row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization
    best_rows = None
    best_val = float('inf')
    
    # Try a few initializations
    for _ in range(5):
        rows = _greedy_init_randomized(n_rows, k, rand_width=3)
        val = max(max(r) for r in rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
            
        if time.time() > start_time + 3.0:
            break

    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    
    # Precompute all_used
    all_used = set()
    for r in current_rows:
        all_used |= _compute_row_diffs(r)
        
    row_maxes = [max(r) for r in current_rows]
    
    # 2. Simulated Annealing / Local Search
    T = 50.0
    T_min = 0.5
    cool = 0.9999
    
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Strategy: Randomly pick a row and try to rebuild it.
        # Also allow standard SA moves for fine-tuning if rebuilds fail.
        
        if random.random() < 0.7:
            # Rebuild move
            ri = random.randrange(n_rows)
            new_row = _rebuild_row_correct(ri, n_rows, k, all_used, current_rows)
            if new_row is not None:
                # Update state
                old_row = current_rows[ri]
                old_diffs = _compute_row_diffs(old_row)
                new_diffs = _compute_row_diffs(new_row)
                
                # Check if valid (should be by construction, but double check)
                # The new_row was built against (all_used - old_diffs).
                # So new_diffs & (all_used - old_diffs) should be empty.
                # And new_diffs are internally distinct.
                
                # Update all_used
                all_used -= old_diffs
                all_used |= new_diffs
                
                current_rows[ri] = new_row
                old_max = row_maxes[ri]
                new_max = max(new_row)
                row_maxes[ri] = new_max
                new_global_max = max(row_maxes)
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    # Accept neutral moves to explore
                    accept = random.random() < 0.5
                else:
                    # Should not happen if we only rebuild if max <= old_max
                    # But if we allowed max > old_max in rebuild, this could happen.
                    # In _rebuild_row_correct, we limited to max_allowed = max(old_row).
                    # So delta should be <= 0.
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                            
                if accept:
                    current_val = new_global_max
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    all_used -= new_diffs
                    all_used |= old_diffs
                    current_rows[ri] = old_row
                    row_maxes[ri] = old_max
                    no_improve += 1
            else:
                no_improve += 1
        else:
            # Standard SA move: Change one element
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
            
            # Try a few random values
            candidates = []
            if old_val > min_allowed:
                candidates.append(old_val - 1)
                if random.random() < 0.5 and old_val - 2 >= min_allowed:
                    candidates.append(old_val - 2)
            if old_val < max_allowed:
                candidates.append(old_val + 1)
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                # Compute diffs involving pi
                old_diffs_pi = set()
                new_diffs_pi = set()
                for i in range(len(current_rows[ri])):
                    if i == pi:
                        continue
                    old_diffs_pi.add(abs(old_val - current_rows[ri][i]))
                    new_diffs_pi.add(abs(new_val - current_rows[ri][i]))
                    
                # Check internal distinctness of new_diffs_pi
                if len(new_diffs_pi) != k:
                    continue
                    
                # Check external
                conflict = False
                for d in new_diffs_pi:
                    if d in all_used and d not in old_diffs_pi:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Valid move
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
                    # Update
                    all_used -= old_diffs_pi
                    all_used |= new_diffs_pi
                    current_rows[ri] = cand_row
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
        
        # Periodic restart if stuck
        if no_improve > 5000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = set()
            for r in current_rows:
                all_used |= _compute_row_diffs(r)
            row_maxes = [max(r) for r in current_rows]
            T = 50.0
            no_improve = 0

    return best_rows

# EVOLVE-BLOCK-END
