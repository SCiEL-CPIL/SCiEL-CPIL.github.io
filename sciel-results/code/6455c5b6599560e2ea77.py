import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    Returns (rows, all_used_bitset)
    """
    rows = []
    used = 0  # Bitset of used differences
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Check if x creates any new difference that is already in 'used'
            # new_diffs are x - a for a in row
            # We check bit by bit? No, we construct the bitset for this row's new diffs and check intersection.
            # However, constructing bitset for every candidate x is expensive.
            # Let's stick to set for init speed, or just simple loop since k is small (6).
            # k=6 means 7 elements. 21 diffs.
            # Let's use set for init for simplicity, then convert to bitset for search.
            pass
            
    # Actually, let's just use a set for init and convert at the end.
    # The speed gain of bitset is in the SA loop.
    rows = []
    used_set = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_set:
                row.append(x)
                used_set |= new_diffs
        rows.append(row)
    
    return rows, used_set

def _set_to_bitset(s):
    """Convert a set of integers to a Python integer bitset."""
    b = 0
    for v in s:
        b |= (1 << v)
    return b

def _bitset_to_set(b):
    """Convert a Python integer bitset to a set of integers."""
    s = set()
    while b:
        # Get index of lowest set bit
        lsb = b & -b
        idx = lsb.bit_length() - 1
        s.add(idx)
        b ^= lsb
    return s

def _row_diffs_bitset(row):
    """Compute the bitset of all positive differences in a row."""
    b = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            b |= (1 << d)
    return b

def _random_greedy_init(n_rows, k, seed):
    """
    Randomized greedy initialization.
    """
    rng = random.Random(seed)
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            candidates = []
            limit = x + 50
            while x <= limit:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates.append(x)
                    if len(candidates) >= 5:
                        break
            if not candidates:
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        break
            
            chosen_x = rng.choice(candidates)
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used |= new_diffs
        rows.append(row)
    
    return rows, used

def _backtracking_row(forbidden_bitset, max_limit, k, rng, max_attempts=50):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_bitset,
    with max entry <= max_limit.
    Uses backtracking to ensure we find a row if one exists, with randomization.
    """
    # We want to find a row [0, x1, x2, ..., xk]
    # x0 = 0
    # Constraints:
    # 1. 0 < x1 < ... < xk <= max_limit
    # 2. All diffs xj - xi are unique within row
    # 3. No diff in row is in forbidden_bitset
    
    def backtrack(idx, current_row, used_in_row_bitset):
        if idx == k + 1:
            return True
        
        last_val = current_row[-1]
        # Try candidates from small to large, but with some randomization
        # To bias towards smaller values (to keep max low), we try smaller ones first
        # but shuffle a bit or pick randomly from a valid set?
        # For backtracking, we need to explore.
        # Let's generate candidates in a random order but biased towards smaller values?
        # Actually, to find *any* valid row under max_limit, order matters less than completeness.
        # But we want compact rows.
        
        # Candidates must be > last_val and <= max_limit
        # Also, for each candidate c, the diffs c - a for a in current_row must be:
        # a) Not in used_in_row_bitset
        # b) Not in forbidden_bitset
        
        # Optimization: Pre-compute valid candidates?
        # Since k is small (6), max_limit is smallish (around 100-200), we can iterate.
        
        # To encourage compactness, we might want to try smaller values first.
        # But pure greedy often fails. Backtracking allows us to "jump" if greedy fails.
        
        # Let's try a mix:
        # 1. Try the smallest valid candidate.
        # 2. If that leads to a dead end, backtrack and try the next.
        
        # To add randomness and avoid getting stuck in the same local minima,
        # we can shuffle the candidate list slightly or pick a random valid candidate
        # from the first N valid ones.
        
        valid_candidates = []
        for c in range(last_val + 1, max_limit + 1):
            # Check diffs
            conflict = False
            for a in current_row:
                d = c - a
                # Check if d is in forbidden or used_in_row
                # Bit check: (forbidden_bitset >> d) & 1
                if (forbidden_bitset >> d) & 1:
                    conflict = True
                    break
                if (used_in_row_bitset >> d) & 1:
                    conflict = True
                    break
            
            if not conflict:
                valid_candidates.append(c)
                
                # Limit candidates to keep branching factor manageable
                if len(valid_candidates) >= max_attempts:
                    break
        
        if not valid_candidates:
            return False
        
        # Randomize the order of candidates to explore different structures
        # But bias towards smaller values?
        # Let's just shuffle. If we always pick smallest, it's greedy.
        # Shuffling allows escaping greedy traps.
        rng.shuffle(valid_candidates)
        
        for c in valid_candidates:
            current_row.append(c)
            # Update used_in_row_bitset
            new_diffs_bitset = 0
            for a in current_row[:-1]:
                d = c - a
                new_diffs_bitset |= (1 << d)
            
            if backtrack(idx + 1, current_row, used_in_row_bitset | new_diffs_bitset):
                return True
            
            current_row.pop()
        
        return False

    current_row = [0]
    # used_in_row starts with 0 diffs
    if backtrack(1, current_row, 0):
        return current_row
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Phase 0: Multi-start Initialization
    best_rows = None
    best_val = float('inf')
    
    # Run a few randomized greedy initializations
    n_starts = 20
    start_budget = 5.0
    for s in range(n_starts):
        if time.time() > start_time + start_budget:
            break
        rows, _ = _random_greedy_init(n_rows, k, seed=s*123 + 42)
        val = max(max(r) for r in rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
            
    if best_rows is None:
        best_rows, _ = _greedy_init(n_rows, k)
        best_val = max(max(r) for r in best_rows)

    # Maintain state for search
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    
    # Pre-compute bitsets
    all_used_bitset = 0
    row_diffs_bitsets = []
    row_maxes = []
    for r in current_rows:
        b = _row_diffs_bitset(r)
        row_diffs_bitsets.append(b)
        all_used_bitset |= b
        row_maxes.append(max(r))
    
    # Phase 1: Row-level reconstruction via Backtracking + SA
    # This allows escaping local minima where single-element moves are stuck
    T = 50.0
    T_min = 1.0
    cool = 0.999
    no_improve = 0
    max_no_improve = 3000
    steps = 0
    
    # Allocate ~40s to Phase 1
    phase1_deadline = start_time + 45.0
    
    while time.time() < phase1_deadline and no_improve < max_no_improve:
        steps += 1
        remaining_time = deadline - time.time()
        if remaining_time < 10.0:
            break
        
        # Pick a row to reconstruct
        # Heuristic: Target rows with high max values
        # 70% chance to pick a row with the current global max
        # 20% chance to pick a row with max close to global max
        # 10% chance random
        if random.random() < 0.7:
            max_val = current_val
            max_rows = [i for i in range(n_rows) if row_maxes[i] == max_val]
            ri = random.choice(max_rows) if max_rows else random.randrange(n_rows)
        elif random.random() < 0.2:
            # Pick from top 3 maxes
            sorted_indices = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
            ri = random.choice(sorted_indices[:3])
        else:
            ri = random.randrange(n_rows)
        
        # Compute forbidden diffs (all diffs from other rows)
        forbidden = 0
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs_bitsets[j]
        
        # Try to find a better row
        # Target: slightly below current global max to encourage improvement
        target_max = current_val - 1
        if target_max < k: # Too small
            target_max = current_rows[ri][-1] - 1
            
        new_row = _backtracking_row(forbidden, target_max, k, random, max_attempts=20)
        
        if new_row is None:
            # Try with current max
            target_max2 = current_val
            new_row = _backtracking_row(forbidden, target_max2, k, random, max_attempts=20)
            
        if new_row is None:
            # Try with slightly larger max for exploration
            target_max3 = current_val + 15
            new_row = _backtracking_row(forbidden, target_max3, k, random, max_attempts=10)
            
        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        new_row_max = new_row[-1]
        
        # Calculate new global max if we replace row ri
        other_maxes = [row_maxes[j] for j in range(n_rows) if j != ri]
        new_global_max = max(max(other_maxes) if other_maxes else 0, new_row_max)
        
        delta = new_global_max - current_val
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if random.random() < 0.5:
                accept = True
        else:
            # Accept worse moves with probability
            if T > T_min:
                prob = math.exp(-delta / max(T, 0.1))
                if random.random() < prob:
                    accept = True
        
        if accept:
            # Update state
            new_diffs_bitset = _row_diffs_bitset(new_row)
            
            # Update all_used_bitset
            # Remove old diffs of ri, add new
            # Note: all_used_bitset is a union. We can't just subtract because other rows might share diffs?
            # NO, the problem states all differences are distinct OVER ALL ROWS.
            # So no diff is shared between rows.
            # Therefore, all_used_bitset = OR of all row_diffs_bitsets.
            # We can recompute or update carefully.
            # Since diffs are unique per row, all_used_bitset & row_diffs_bitsets[ri] == row_diffs_bitsets[ri]
            # So: all_used_bitset = (all_used_bitset & ~row_diffs_bitsets[ri]) | new_diffs_bitset
            # But ~ in Python is infinite precision.
            # Better: all_used_bitset = (all_used_bitset ^ row_diffs_bitsets[ri]) | new_diffs_bitset
            # Wait, if we XOR, we remove the bits. Then OR adds new ones.
            # This works if new_diffs_bitset has no overlap with (all_used - old). Which is true by construction (forbidden check).
            
            all_used_bitset = (all_used_bitset ^ row_diffs_bitsets[ri]) | new_diffs_bitset
            
            current_rows[ri] = new_row
            row_diffs_bitsets[ri] = new_diffs_bitset
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
            
        T = max(T_min, T * cool)
        
        # Periodic restart if stuck
        if no_improve > max_no_improve // 2:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            # Recompute bitsets
            all_used_bitset = 0
            row_diffs_bitsets = []
            row_maxes = []
            for r in current_rows:
                b = _row_diffs_bitset(r)
                row_diffs_bitsets.append(b)
                all_used_bitset |= b
                row_maxes.append(max(r))
            T = 50.0
            no_improve = 0

    # Phase 2: Element-level simulated annealing for fine-tuning
    # Allocate ~50s to Phase 2
    phase2_deadline = min(deadline, start_time + 95.0)
    T = 10.0
    T_min = 0.5
    cool = 0.99995
    no_improve = 0
    max_no_improve_polish = 20000
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used_bitset = 0
    row_diffs_bitsets = []
    row_maxes = []
    for r in current_rows:
        b = _row_diffs_bitset(r)
        row_diffs_bitsets.append(b)
        all_used_bitset |= b
        row_maxes.append(max(r))
    
    while time.time() < phase2_deadline and no_improve < max_no_improve_polish:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        # Decreasing candidates (primary goal)
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing candidates (for exploration, limited)
        if pi < k:
            max_allowed = current_rows[ri][pi + 1] - 1
        else:
            max_allowed = current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        # Random jump
        if random.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            # Basic validity check for this position
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Build candidate row
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Calculate new diffs for this row efficiently
            n = len(cand_row)
            new_diffs_bitset = 0
            valid_internal = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    # Check if d is already in new_diffs_bitset
                    if (new_diffs_bitset >> d) & 1:
                        valid_internal = False
                        break
                    new_diffs_bitset |= (1 << d)
                if not valid_internal:
                    break
            
            if not valid_internal:
                continue
                
            # Get old diffs of row ri
            old_diffs_bitset = row_diffs_bitsets[ri]
            
            # Check for conflicts: new_diffs should not intersect with (all_used - old_diffs)
            # conflict_bits = new_diffs_bitset & (all_used_bitset ^ old_diffs_bitset)
            # Note: all_used_bitset ^ old_diffs_bitset removes the old diffs from the global set.
            # If new_diffs has any bit in that remaining set, it's a conflict.
            conflict_bits = new_diffs_bitset & (all_used_bitset ^ old_diffs_bitset)
            
            if conflict_bits:
                continue
                
            # Move is valid - compute new global max efficiently
            cand_row_max = max(cand_row)
            
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
                if random.random() < 0.2:
                    accept = True
            else:
                # Accept worse moves with probability
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used_bitset = (all_used_bitset ^ old_diffs_bitset) | new_diffs_bitset
                current_rows[ri] = cand_row
                row_diffs_bitsets[ri] = new_diffs_bitset
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
        if no_improve > max_no_improve_polish // 2:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used_bitset = 0
            row_diffs_bitsets = []
            row_maxes = []
            for r in current_rows:
                b = _row_diffs_bitset(r)
                row_diffs_bitsets.append(b)
                all_used_bitset |= b
                row_maxes.append(max(r))
            T = 10.0
            no_improve = 0
    
    # Phase 3: Final polishing with only decreasing moves
    final_deadline = deadline
    T = 1.0
    T_min = 0.1
    cool = 0.99999
    no_improve = 0
    max_no_improve_final = 5000
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used_bitset = 0
    row_diffs_bitsets = []
    row_maxes = []
    for r in current_rows:
        b = _row_diffs_bitset(r)
        row_diffs_bitsets.append(b)
        all_used_bitset |= b
        row_maxes.append(max(r))
    
    while time.time() < final_deadline and no_improve < max_no_improve_final:
        # Focus on the row(s) with the maximum value
        max_val = current_val
        max_rows = [i for i in range(n_rows) if row_maxes[i] == max_val]
        
        if not max_rows:
            break
            
        ri = random.choice(max_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            n = len(cand_row)
            new_diffs_bitset = 0
            valid_internal = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if (new_diffs_bitset >> d) & 1:
                        valid_internal = False
                        break
                    new_diffs_bitset |= (1 << d)
                if not valid_internal:
                    break
            
            if not valid_internal:
                continue
                
            old_diffs_bitset = row_diffs_bitsets[ri]
            
            conflict_bits = new_diffs_bitset & (all_used_bitset ^ old_diffs_bitset)
            
            if conflict_bits:
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
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.05))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used_bitset = (all_used_bitset ^ old_diffs_bitset) | new_diffs_bitset
                current_rows[ri] = cand_row
                row_diffs_bitsets[ri] = new_diffs_bitset
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

    return best_rows

# EVOLVE-BLOCK-END
