import time
import random
import math

def _scope(rows):
    """Calculate the scope (maximum entry) of the rows."""
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    """Check if the rows form a valid difference triangle set."""
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        # Check strictly increasing
        for i in range(len(r) - 1):
            if r[i] >= r[i+1]:
                return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_construct(n_rows, k, rng, bias=1.0):
    """
    Construct a valid set of rows using a greedy approach.
    bias influences how aggressively we pick smaller numbers.
    """
    used_diffs = 0  # Bitmask of used differences
    max_diff = 256  # Initial guess, will expand if needed
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        last = 0
        for _ in range(k):
            found = False
            # Try values starting from last + 1
            x = last + 1
            while not found:
                # Check if x causes any collision
                valid = True
                new_diffs_mask = 0
                for v in row:
                    d = x - v
                    if d >= max_diff:
                        # Expand bitmask if necessary (rare if max_diff is large enough)
                        # For simplicity, we assume max_diff covers the scope.
                        # If scope is ~112, max_diff 256 is safe.
                        pass
                    if (used_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    row.append(x)
                    last = x
                    used_diffs |= new_diffs_mask
                    found = True
                else:
                    x += 1
                    # Safety break to avoid infinite loops if something is wrong
                    if x > 1000:
                        break
        rows.append(row)
    return rows

def _sa_optimize(rows, k, deadline, rng):
    """
    Simulated Annealing optimization using bitmask difference tracking.
    """
    n_rows = len(rows)
    # Deep copy
    cur_rows = [list(r) for r in rows]
    
    # Precompute difference masks for each element in each row
    # mask[i][j] is the bitmask of differences contributed by element j in row i
    # to the global set of differences.
    # Note: A difference d between a and b is contributed by both a and b in the sense that 
    # if we remove a, we remove d. If we change a, we remove old d's and add new d's.
    
    # To make updates fast, we maintain:
    # 1. global_mask: The union of all differences.
    # 2. For each position (i, j), we need to know which differences are "unique" to it 
    #    or shared. Actually, it's easier to just recompute the contribution of the row 
    #    if we change an element, but that's O(k^2). 
    #    Better: Maintain global_mask. 
    #    When changing rows[i][j] from old to new:
    #    1. Compute mask_old: differences between old and all other elements in row i.
    #    2. Compute mask_new: differences between new and all other elements in row i.
    #    3. The differences that ONLY existed because of 'old' are (mask_old & ~global_mask) 
    #       ... wait, this is tricky because differences might be shared between pairs in the same row.
    #       Example: row [0, 1, 2]. Diffs: 1, 2, 1. 
    #       If we change 1 to 3: row [0, 3, 2]. Diffs: 3, 2, 1.
    #       
    #    Let's use a simpler but slightly slower approach for correctness and speed balance:
    #    Maintain a count array for differences? No, bitmask is better for "is it used".
    #    But we need to know if a difference is used by ANY other pair.
    #    
    #    Strategy:
    #    Maintain global_mask (bit d is 1 if diff d is present anywhere).
    #    When moving element (i, j):
    #    1. Identify all differences involving (i, j) in the current state. Let this set be S_old.
    #    2. For each d in S_old, we need to know if d is used by any OTHER pair.
    #       This requires knowing the "multiplicity" of each difference.
    #       
    #    Alternative: 
    #    Since k is small (6), the number of differences per row is 15.
    #    Total differences is 7 * 15 = 105.
    #    We can maintain a dictionary: diff -> count of pairs producing it.
    #    This is O(1) to update if we iterate through the pairs involving the moved element.
    #    
    #    Let's switch to a Counter/dict approach for counts, but use bitmask for quick "is empty" checks if needed?
    #    Actually, a dict of counts is very fast in Python for small sizes.
    
    # Initialize counts
    diff_counts = {}
    for r in cur_rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
    
    scope = _scope(cur_rows)
    best_scope = scope
    best_rows = [list(r) for r in cur_rows]
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute indices for faster access
    # For a move at (i, j), the affected pairs are (i, t) for all t != j.
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k) # Don't move the 0
        
        row = cur_rows[i]
        old_val = row[j]
        
        # Determine range for new value
        lo = row[j-1] + 1
        hi = row[j+1] - 1 if j < k else scope + 100
        
        if lo > hi:
            continue
            
        # Propose new value
        if rng.random() < 0.7 and old_val > lo:
            # Try to reduce
            max_drop = old_val - lo
            drop = rng.randint(1, max(1, max_drop // 2))
            new_val = old_val - drop
        else:
            # Random perturbation
            span = hi - lo
            if span <= 0: continue
            delta = int(rng.gauss(0, max(1, span * 0.2)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
            
        if new_val == old_val:
            continue
            
        # Calculate cost change
        # 1. Remove old differences involving (i, j)
        # 2. Add new differences involving (i, j)
        # 3. Check for collisions (count > 1)
        
        # We need to update diff_counts temporarily
        # To avoid modifying the main state if we reject, we can just calculate the delta.
        
        # Pairs involving j in row i:
        # (i, 0) ... (i, j-1), (i, j+1) ... (i, k)
        
        # Let's build the list of (old_diff, new_diff) pairs
        # Note: The value at j changes from old_val to new_val.
        # The differences are |new_val - row[t]|.
        
        # We need to know if any new_diff is already present in the *rest* of the system.
        # "Rest of system" = (Global Counts) - (Counts of old diffs involving j) + (Counts of new diffs involving j)?
        # No.
        
        # Let's do a trial update on a copy of the counts? 
        # Dict copy is O(N_diffs) ~ 100 operations. That's fine.
        
        trial_counts = dict(diff_counts)
        
        # Remove old
        for t in range(k + 1):
            if t == j: continue
            d = abs(row[t] - old_val)
            trial_counts[d] -= 1
            if trial_counts[d] == 0:
                del trial_counts[d]
                
        # Add new
        valid = True
        for t in range(k + 1):
            if t == j: continue
            d = abs(row[t] - new_val)
            if d in trial_counts:
                # Collision!
                valid = False
                break
            else:
                trial_counts[d] = 1
                # We don't need to add to trial_counts fully if we already found a collision,
                # but if valid so far, we just check existence.
        
        if not valid:
            # High temperature might accept collisions? 
            # No, the problem requires distinct differences. 
            # We can only accept moves that result in a VALID state.
            # So we only consider valid moves.
            # However, if we restrict to valid moves, we might get stuck.
            # Standard SA for CSPs often allows invalid states with a penalty.
            # Let's stick to valid moves only for simplicity and robustness, 
            # but use "restarts" to escape local minima.
            continue
            
        # If valid, calculate new scope
        # The scope is the max value in all rows.
        # If old_val was the max, and new_val < old_val, scope might decrease.
        # If new_val > old_val, scope increases.
        
        # Check if old_val was the global max
        # And if new_val is the new global max
        
        # Optimization: 
        # If new_val > scope: new_scope = new_val
        # Else if old_val == scope: 
        #    new_scope = max(max of row i without j, max of other rows)
        #    This is expensive. 
        #    Approximation: 
        #    If old_val == scope and new_val < scope:
        #       We need to find the next largest value.
        #       We can just recompute scope if we accept.
        #       For cost evaluation, we can estimate.
        
        # Let's just compute new_scope if the move is likely to be accepted or if it changes the max.
        # To save time, let's just assume cost = 0 for valid moves, and use "restarts" + "greedy refinement" for scope reduction.
        # Actually, we want to MINIMIZE scope.
        # So we should prefer moves that reduce the max element.
        
        # Let's define cost = scope.
        # If we don't know the new scope exactly, we can approximate.
        
        # Quick check:
        if new_val > scope:
            new_scope = new_val
        elif old_val == scope:
            # Need to find max of the rest
            # Max of row i (excluding j)
            max_row_i = 0
            for t in range(k + 1):
                if t != j and row[t] > max_row_i:
                    max_row_i = row[t]
            # Max of other rows
            max_other = 0
            for r_idx in range(n_rows):
                if r_idx != i:
                    if cur_rows[r_idx][-1] > max_other:
                        max_other = cur_rows[r_idx][-1]
            new_scope = max(max_row_i, max_other)
        else:
            new_scope = scope
            
        # Acceptance criterion
        # We want to minimize scope.
        # If new_scope < scope: Always accept (or with high prob)
        # If new_scope == scope: Accept with prob exp(-0/T) = 1? No, that's random walk.
        # We should have a slight bias towards lower values or just random walk within valid states.
        # To escape local minima, we accept equal cost moves with prob 1 (random walk) or slightly less.
        # To reduce scope, we accept improvements.
        
        if new_scope < scope:
            # Accept
            cur_rows[i][j] = new_val
            diff_counts = trial_counts
            scope = new_scope
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in cur_rows]
            # Reheat
            T = min(T * 1.1, 15.0)
        else:
            # new_scope >= scope
            # Accept with probability exp(-(new_scope - scope) / T)
            # Since new_scope >= scope, delta >= 0.
            delta = new_scope - scope
            if delta == 0:
                # Random walk
                if rng.random() < 0.5: # Accept half the time to explore
                    cur_rows[i][j] = new_val
                    diff_counts = trial_counts
                    # scope unchanged
            else:
                # Worse scope
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        cur_rows[i][j] = new_val
                        diff_counts = trial_counts
                        scope = new_scope
        
        T = max(T * cooling, min_T)
        
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple Greedy Starts
    # We run a few greedy constructions with different random seeds/orders
    # and pick the best one.
    
    n_restarts = 20
    phase1_deadline = start_time + 10.0
    
    for i in range(n_restarts):
        if time.time() > phase1_deadline:
            break
        
        # Vary the bias in greedy
        # The greedy function above doesn't take bias, let's just run it.
        # To vary it, we can shuffle the order of rows or just use different RNGs.
        # The greedy is deterministic given the RNG.
        
        rows = _greedy_construct(n_rows, k, rng)
        
        if _valid(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback: pure sequential greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            for _ in range(k):
                x = last + 1
                while True:
                    valid = True
                    for v in row:
                        d = x - v
                        if d in used:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        last = x
                        used.update(x - v for v in row[:-1])
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
        
    # Phase 2: Simulated Annealing
    # Run SA for the remaining time.
    # We can run multiple short SA sessions from the best start found.
    
    remaining = deadline - time.time()
    if remaining > 5:
        # Run SA
        # We'll run it until deadline - 5s
        sa_deadline = time.time() + remaining - 5.0
        sa_scope, sa_rows = _sa_optimize(best_rows, k, sa_deadline, rng)
        
        if _valid(sa_rows, k):
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
                
    # Final check
    if not _valid(best_rows, k):
        # Emergency fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            for _ in range(k):
                x = last + 1
                while True:
                    valid = True
                    for v in row:
                        d = x - v
                        if d in used:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        last = x
                        used.update(x - v for v in row[:-1])
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        
    return best_rows
