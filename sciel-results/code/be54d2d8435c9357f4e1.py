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
        if d in all_used and d not in old_diffs:
            return None, False
            
        new_diffs.add(d)
        
    return new_diffs, True

def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Greedy local search to minimize the global maximum.
    Iterates through rows and positions, trying to decrease each value as much as possible.
    """
    current_rows = [r[:] for r in rows]
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    current_val = max(row_maxes)
    best_val = current_val
    best_rows = [r[:] for r in current_rows]
    
    # We will perform multiple passes. In each pass, we try to improve any position.
    # We stop if we reach the deadline or if a full pass yields no improvement.
    pass_count = 0
    max_passes = 1000
    
    while time.time() < deadline and pass_count < max_passes:
        improved_in_pass = False
        pass_count += 1
        
        # Randomize order of (row, pos) to avoid bias
        indices = [(r, p) for r in range(n_rows) for p in range(1, k + 1)]
        random.shuffle(indices)
        
        for ri, pi in indices:
            if time.time() > deadline:
                break
                
            row = current_rows[ri]
            old_val = row[pi]
            
            # We want to find the smallest valid value for this position
            # that is greater than row[pi-1] and less than row[pi+1] (if exists)
            lower_bound = row[pi - 1] + 1
            upper_bound = old_val - 1
            if pi < k:
                upper_bound = min(upper_bound, row[pi + 1] - 1)
            
            if upper_bound < lower_bound:
                continue
                
            # Try values from smallest to largest? Or largest decrease first?
            # To minimize scope, we want the smallest possible value.
            # However, a smaller value might block other reductions.
            # Greedy: try to find ANY valid value smaller than old_val.
            # If we find one, take the smallest one? Or just the first one?
            # Taking the smallest one is aggressive and good for scope minimization.
            
            # Optimization: Instead of checking every value, we can scan from lower_bound upwards
            # and take the first valid one.
            found = False
            # Limit search depth to avoid getting stuck if no small value exists
            # But we want to go as low as possible.
            for new_val in range(lower_bound, upper_bound + 1):
                if new_val >= old_val:
                    break
                    
                static_pi = static_diffs[ri][pi]
                old_diffs = row_diffs[ri]
                
                new_diffs, valid = _check_move_validity(
                    current_rows[ri], pi, new_val, static_pi, all_used, old_diffs)
                
                if valid:
                    # Update state
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri][pi] = new_val
                    row_diffs[ri] = new_diffs
                    static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                    
                    # Update row max
                    if pi == k:
                        row_maxes[ri] = new_val
                    # Note: if pi < k, row max doesn't change unless new_val > old row max, 
                    # but new_val < old_val <= old row max, so row max is unchanged.
                    
                    # Update global max
                    new_global_max = max(row_maxes)
                    if new_global_max < current_val:
                        current_val = new_global_max
                        improved_in_pass = True
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                    found = True
                    break # Move to next position/row
                    
            # If we didn't find a smaller value, we leave it as is.
            # We do not try to increase values in this greedy phase.
        
        if not improved_in_pass:
            # If no improvement in a full pass, we might be stuck.
            # Break to allow ILS to perturb.
            break
            
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    # Try a few greedy initializations to see if one is significantly better
    best_rows = None
    best_val = float('inf')
    
    # Standard greedy
    rows_greedy = _greedy_init(n_rows, k)
    val_greedy = _global_max(rows_greedy)
    best_rows = rows_greedy
    best_val = val_greedy
    
    # 2. Iterated Local Search (ILS)
    # We will run ILS cycles until deadline.
    # Each cycle:
    # a) Take current best
    # b) Perturb (random valid moves)
    # c) Local Search
    # d) Accept if better
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    
    # We will keep a pool of "diverse" solutions to perturb from, not just the best.
    # But for simplicity and focus on "structural change", we'll stick to perturbing the best
    # with a mechanism that allows escaping local minima.
    
    # To escape local minima, we can occasionally accept worse solutions in the ILS loop
    # or use a "restart" strategy where we re-initialize if stuck for too long.
    
    stagnation_counter = 0
    max_stagnation = 50 # Number of ILS cycles without improvement before restart
    
    while time.time() < deadline:
        # 1. Perturb
        perturbed_rows = [r[:] for r in current_rows]
        
        # Perform a few random valid moves
        n_perturb = random.randint(3, 8)
        for _ in range(n_perturb):
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            row = perturbed_rows[ri]
            old_val = row[pi]
            
            # Find a random valid value different from old_val
            # We need to compute diffs for this specific state
            # This is expensive if done naively, but n_perturb is small.
            # Let's use a helper to find a random valid value.
            
            # Compute all_used for perturbed state? No, that's O(N^2).
            # Instead, we can just try to move pi to a random nearby value and check validity.
            # If valid, accept.
            
            # Define search range
            low = row[pi - 1] + 1
            high = (row[pi + 1] - 1) if pi < k else old_val + 20
            
            # Try a few random candidates
            found_move = False
            for _ in range(10):
                if low > high:
                    break
                new_val = random.randint(low, high)
                if new_val == old_val:
                    continue
                    
                # Check validity against current perturbed state
                # We need static diffs and all_used for perturbed_rows
                # This is getting expensive. 
                # Alternative: Perturb by swapping or shifting?
                # Let's stick to simple random moves but compute validity efficiently.
                
                # To make this fast, we can just rebuild the state if we accept.
                # But checking validity requires all_used.
                # Let's compute all_used once per perturbation step? No, too slow.
                
                # Strategy: 
                # 1. Compute all_used for current current_rows (best).
                # 2. Perturb by picking a random (ri, pi) and a random new_val.
                # 3. Check if move is valid in current_rows.
                # 4. If valid, apply to perturbed_rows.
                # 5. If perturbed_rows becomes invalid (due to previous perturbations), 
                #    we need to track diffs.
                
                # This is complex. Let's simplify:
                # Perturbation: Pick a random row and a random position. 
                # Try to move it to a random valid position *in the context of the original best solution*.
                # If valid, apply. This creates a solution that is "close" to best.
                # Then run local search.
                
                # Let's pre-compute all_used for current_rows (best) outside the loop.
                # But we are inside the loop.
                
                # Let's restructure:
                # We will maintain all_used, row_diffs, static_diffs for current_rows.
                # Perturbation: 
                #   Pick ri, pi.
                #   Pick new_val randomly in range.
                #   Check validity against current_rows state.
                #   If valid, update perturbed_rows and a temporary state tracker?
                #   Actually, if we only do a few moves, we can just recompute state for perturbed_rows
                #   before local search.
                
                pass # This inner logic is messy. Let's use a simpler perturbation:
                # Just pick a random (ri, pi) and set it to a random valid value 
                # found by scanning a small range, checking against a freshly computed 
                # all_used of the *current best* (not perturbed). 
                # This ensures the perturbed solution is valid if only one move is made.
                # If multiple moves are made, they might conflict.
                # So, limit to 1-2 moves, or check validity incrementally.
                
                # Let's just do 1 move per perturbation for simplicity and speed.
                # And we will check validity against the *original* current_rows state.
                
                # Wait, if we do 1 move, the result is valid if the move is valid in the original state.
                # So:
                # 1. Compute all_used for current_rows.
                # 2. Pick ri, pi.
                # 3. Find a random valid new_val for (ri, pi) in current_rows.
                # 4. If found, perturbed_rows[ri][pi] = new_val.
                # 5. Break perturbation loop.
                
                # Let's implement this properly.
                break # End the inner random attempts, we'll do this outside the loop for clarity
        
        # Better Perturbation Logic:
        all_used_best = _compute_all_used(current_rows)
        row_diffs_best = [_row_diffs_set(r) for r in current_rows]
        static_diffs_best = [_build_static_diffs(r, k) for r in current_rows]
        
        # Try to make 1-3 valid moves
        n_moves = random.randint(1, 3)
        for _ in range(n_moves):
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            row = current_rows[ri]
            old_val = row[pi]
            low = row[pi - 1] + 1
            high = (row[pi + 1] - 1) if pi < k else old_val + 50
            
            if low > high:
                continue
                
            # Sample random candidates
            candidates = []
            if high - low < 20:
                candidates = list(range(low, high + 1))
            else:
                candidates = [random.randint(low, high) for _ in range(10)]
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val == old_val:
                    continue
                new_diffs, valid = _check_move_validity(
                    current_rows[ri], pi, new_val, static_diffs_best[ri][pi], all_used_best, row_diffs_best[ri])
                if valid:
                    # Apply to perturbed_rows
                    perturbed_rows[ri][pi] = new_val
                    # Note: We don't update all_used_best here because subsequent moves
                    # in this perturbation phase should ideally be checked against the
                    # *new* state. But for simplicity and speed, we assume moves are
                    # independent or check against the original state.
                    # This might lead to invalid perturbed states if moves conflict.
                    # To be safe, we should update the tracking sets.
                    
                    # Let's update tracking sets for the perturbed state
                    all_used_best -= row_diffs_best[ri]
                    all_used_best |= new_diffs
                    row_diffs_best[ri] = new_diffs
                    static_diffs_best[ri] = _build_static_diffs(perturbed_rows[ri], k)
                    
                    # Update current_rows state for future moves?
                    # No, perturbed_rows is separate.
                    # But we used current_rows for the check.
                    # We need to check against the *current perturbed* state.
                    # This is getting complicated.
                    
                    # Simplification: Only do 1 move per perturbation.
                    moved = True
                    break
            
            if moved:
                break # Only 1 move
        
        # 2. Local Search on Perturbed
        improved_rows, improved_val = _local_search_minimize(perturbed_rows, n_rows, k, deadline)
        
        # 3. Acceptance
        if improved_val < best_val:
            best_val = improved_val
            best_rows = improved_rows
            current_rows = improved_rows
            current_val = improved_val
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
            # If stuck for too long, restart from a fresh greedy or random restart
            if stagnation_counter > max_stagnation:
                # Restart
                current_rows = _greedy_init(n_rows, k)
                current_val = _global_max(current_rows)
                # We don't update best_rows here, we just reset the "current" for ILS
                stagnation_counter = 0
                # Optionally, if greedy is worse than best, we should probably
                # restart from best but with a stronger perturbation.
                # For now, let's just try to improve the greedy.
                # If greedy is much worse, local search might not catch up.
                # So, let's restart from best_rows but with a larger perturbation.
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                
                # Strong Perturbation
                for _ in range(random.randint(10, 20)):
                    ri = random.randrange(n_rows)
                    pi = random.randrange(1, k + 1)
                    row = current_rows[ri]
                    old_val = row[pi]
                    low = row[pi - 1] + 1
                    high = (row[pi + 1] - 1) if pi < k else old_val + 50
                    if low <= high:
                        new_val = random.randint(low, high)
                        # Check validity
                        # We need state for current_rows
                        # This is expensive. Let's just set it and hope local search fixes it?
                        # No, local search assumes valid input.
                        # So we must ensure validity.
                        # Let's skip strong perturbation if it's too hard to validate quickly.
                        pass

    return best_rows

# EVOLVE-BLOCK-END
