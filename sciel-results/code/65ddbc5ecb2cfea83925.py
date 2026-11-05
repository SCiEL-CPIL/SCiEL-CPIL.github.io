import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for safety

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def calc_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row(used, k, strategy="pure", rng=None, max_scope_hint=None):
        """
        Build one row greedily.
        used: set of forbidden differences.
        strategy: 'pure', 'shuffled', 'lookahead'
        """
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            if strategy == "pure":
                x += 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used |= new_diffs
                found = True

            elif strategy == "shuffled":
                # Try a window of candidates
                base = x + 1
                window_size = 30
                candidates = list(range(base, base + window_size))
                if rng:
                    rng.shuffle(candidates)
                
                # Ensure we don't skip too far if window fails, but for greedy init, 
                # we can just scan forward if window fails
                for cand in candidates:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        row.append(cand)
                        used |= new_diffs
                        x = cand
                        found = True
                        break
                
                if not found:
                    # Fallback to pure scan from base
                    cand = base
                    while True:
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            row.append(cand)
                            used |= new_diffs
                            x = cand
                            break
                        cand += 1

            elif strategy == "lookahead":
                # Find valid candidates in a window, pick one that allows future growth
                base = x + 1
                window_size = 50
                valid_cands = []
                for cand in range(base, base + window_size):
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        valid_cands.append(cand)
                    if len(valid_cands) >= 5:
                        break
                
                if not valid_cands:
                    # Fallback
                    cand = base
                    while True:
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            valid_cands.append(cand)
                            break
                        cand += 1

                best_cand = valid_cands[0]
                best_score = float('inf')
                
                for cand in valid_cands:
                    # Heuristic: prefer smaller cand, but also check if it blocks immediate next steps
                    # Simple score: cand value
                    score = cand
                    # Optional: count how many diffs it uses that are "small" (might be more constrained?)
                    # For now, just smallest is usually best for initial packing
                    if score < best_score:
                        best_score = score
                        best_cand = cand

                new_diffs = {best_cand - a for a in row}
                row.append(best_cand)
                used |= new_diffs
                x = best_cand
                found = True

            if not found:
                return None

        return row

    def backtrack_row(used_forbidden, k, max_scope, rng=None):
        """
        Attempt to build a row of length k+1 starting at 0, strictly increasing,
        max value <= max_scope, and no differences in used_forbidden.
        Returns row or None.
        """
        row = [0]
        
        # Optimization: Pre-compute available numbers? 
        # For small k, backtracking is fast enough.
        
        def bt():
            if len(row) == k + 1:
                return True
            
            last_val = row[-1]
            # Next value must be > last_val and <= max_scope
            # Also, the difference (next_val - last_val) must not be in used_forbidden
            # And all differences (next_val - a) for a in row must not be in used_forbidden
            
            start = last_val + 1
            end = max_scope
            
            # Heuristic: try smaller values first to keep scope low
            # If rng is provided and we want diversity, we could shuffle, 
            # but for "rebuild to reduce scope", we want the *tightest* fit, so ordered is better.
            
            for x in range(start, end + 1):
                # Check differences
                valid = True
                for a in row:
                    d = x - a
                    if d in used_forbidden:
                        valid = False
                        break
                
                if valid:
                    row.append(x)
                    # Note: We don't add to used_forbidden here because we are building 
                    # against the *other* rows' diffs. Internal diffs are fine as long as 
                    # they don't collide with *other* rows. 
                    # Wait, the problem says "differences ... are distinct over all rows".
                    # This means no diff in row A can equal any diff in row B.
                    # Diffs within the same row are distinct by definition of the set? 
                    # No, "all differences between two entries of the same row are distinct over all rows".
                    # This implies:
                    # 1. Within a row, all diffs are distinct? 
                    #    "differences between two entries of the same row are distinct" -> Yes, standard Sidon-like property per row?
                    #    Actually, if a row is {0, 2, 5}, diffs are 2, 5, 3. All distinct.
                    #    If a row is {0, 1, 3}, diffs are 1, 3, 2. All distinct.
                    #    Is it possible to have a row where diffs are NOT distinct?
                    #    {0, 1, 2} -> diffs 1, 2, 1. Duplicate 1.
                    #    The prompt says: "all differences between two entries of the same row are distinct over all rows".
                    #    This phrasing usually implies the set of ALL diffs from ALL rows is a set (no duplicates).
                    #    So, within a row, diffs must be distinct, AND across rows, diffs must be distinct.
                    
                    # So, when building a row, we must ensure internal diffs are distinct too?
                    # If we build strictly increasing, internal diffs are distinct?
                    # No. {0, 1, 3} -> 1, 2, 3. Distinct.
                    # {0, 2, 4} -> 2, 2, 4. Duplicate 2.
                    # So we MUST check internal distinctness as well.
                    
                    # However, the `used_forbidden` set contains diffs from OTHER rows.
                    # We also need to track internal diffs to ensure they don't collide with each other.
                    
                    # Let's refine:
                    # We need a set `internal_diffs` for the current row being built.
                    # And we need to check against `used_forbidden` (other rows).
                    
                    # My current bt() logic only checks against `used_forbidden`.
                    # I need to also check against `internal_diffs`.
                    
                    # Let's restart the bt logic with internal tracking.
                    
                    # To keep it fast, I will pass `internal_diffs` as a closure variable or arg.
                    # Since this is a nested function, I'll use a nonlocal or just a list in the outer scope of bt.
                    
                    # Actually, let's just return False and let the caller handle it, 
                    # or better, fix the bt function.
                    
                    # For now, assume the caller ensures we check internal diffs.
                    # I will modify the bt function below.
                    
                    pass # Placeholder to indicate logic change needed in implementation below

        # Corrected BT implementation
        internal_diffs = set()
        
        def bt_correct():
            if len(row) == k + 1:
                return True
            
            last_val = row[-1]
            start = last_val + 1
            end = max_scope
            
            for x in range(start, end + 1):
                valid = True
                new_internal_diffs = set()
                for a in row:
                    d = x - a
                    if d in used_forbidden or d in internal_diffs:
                        valid = False
                        break
                    new_internal_diffs.add(d)
                
                if valid:
                    row.append(x)
                    internal_diffs.update(new_internal_diffs)
                    
                    if bt_correct():
                        return True
                    
                    row.pop()
                    internal_diffs.difference_update(new_internal_diffs)
            
            return False
        
        if bt_correct():
            return row
        return None

    # --- Initialization ---
    
    # Try a few diverse constructions to find a good starting point
    best_init_rows = None
    best_init_scope = float('inf')
    
    strategies = ["pure", "shuffled", "lookahead"]
    for _ in range(5): # 5 restarts
        for strat in strategies:
            if time.time() > deadline - 135:
                break
            rng = random.Random()
            used = set()
            rows = []
            success = True
            for i in range(n_rows):
                r = greedy_row(used, k, strat, rng)
                if r is None:
                    success = False
                    break
                rows.append(r)
            if success:
                scope = get_scope(rows)
                if scope < best_init_scope:
                    best_init_scope = scope
                    best_init_rows = [r[:] for r in rows]

    if best_init_rows is None:
        # Fallback: very loose greedy
        used = set()
        rows = []
        for i in range(n_rows):
            r = greedy_row(used, k, "pure")
            if r is None:
                # Should not happen
                r = list(range(k + 1))
            rows.append(r)
        best_init_rows = rows
        best_init_scope = get_scope(rows)

    rows = best_init_rows
    current_scope = best_init_scope
    
    # --- Main Optimization Loop ---
    
    # Pre-compute diffs for all rows
    row_diffs = [calc_row_diffs(r) for r in rows]
    used_diffs = set()
    for d_set in row_diffs:
        used_diffs |= d_set
    
    t_end = deadline - 5.0 # Leave 5s for polish
    initial_temp = max(1.0, current_scope / 5.0)
    cooling_rate = 0.999
    steps = 0
    
    # Probability of doing a "Rebuild" vs "Element Move"
    # Rebuild is expensive but powerful. Start with low prob, increase if stuck?
    # Or just mix them.
    p_rebuild = 0.1
    
    while time.time() < t_end:
        row_idx = random.randint(0, n_rows - 1)
        
        move_type = "rebuild" if random.random() < p_rebuild else "element"
        
        if move_type == "rebuild":
            # Try to rebuild row_idx with a slightly smaller scope
            # Target scope: current_scope - 1 or current row max - 1
            target_scope = current_scope - 1
            if target_scope < k + 1:
                target_scope = k + 1
            
            # Calculate forbidden diffs (all diffs except row_idx)
            forbidden = set()
            for i in range(n_rows):
                if i != row_idx:
                    forbidden |= row_diffs[i]
            
            # Attempt rebuild
            new_row = backtrack_row(forbidden, k, target_scope)
            
            if new_row is not None:
                new_row_diffs = calc_row_diffs(new_row)
                new_scope = max(max(new_row), max(max(rows[i]) for i in range(n_rows) if i != row_idx))
                
                # Accept if it improves scope or with SA probability
                if new_scope < current_scope:
                    # Update state
                    used_diffs.difference_update(row_diffs[row_idx])
                    used_diffs.update(new_row_diffs)
                    
                    rows[row_idx] = new_row
                    row_diffs[row_idx] = new_row_diffs
                    current_scope = new_scope
                else:
                    # If same scope, maybe accept to escape local min?
                    # But we need to ensure validity. Since we built against forbidden, it is valid.
                    # Accept with SA prob
                    if random.random() < pow(2.718281828, (current_scope - new_scope) / initial_temp):
                        used_diffs.difference_update(row_diffs[row_idx])
                        used_diffs.update(new_row_diffs)
                        rows[row_idx] = new_row
                        row_diffs[row_idx] = new_row_diffs
                        current_scope = new_scope
            
        else:
            # Element Move
            pos = random.randint(1, k)
            old_val = rows[row_idx][pos]
            
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else current_scope + 10
            
            if min_val > max_val:
                initial_temp *= cooling_rate
                continue
            
            # Try a few random candidates
            accepted = False
            for _ in range(3):
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue
                
                # Remove old diffs involving pos
                old_diffs = set()
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = old_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - old_val
                        old_diffs.add(d)
                
                # Check if new_val creates collisions
                new_diffs = set()
                valid = True
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - new_val
                        # Check against global used_diffs (excluding old_diffs of this row)
                        # Note: used_diffs contains ALL diffs.
                        # We need to check if d is in (used_diffs - old_diffs)
                        if d in used_diffs and d not in old_diffs:
                            valid = False
                            break
                        # Also check internal collision with other new diffs?
                        # Since we are replacing one element, the other elements are fixed.
                        # The new diffs are with fixed elements.
                        # We just need to ensure they don't collide with each other?
                        # No, if new_val is distinct from others, and others are distinct,
                        # the diffs new_val - a_i are distinct?
                        # Not necessarily. 
                        # But if new_val - a_i == new_val - a_j, then a_i == a_j, which is false.
                        # So internal distinctness is guaranteed if values are distinct.
                        new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Calculate new scope
                new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
                other_max = 0
                for i in range(n_rows):
                    if i != row_idx:
                        if rows[i][-1] > other_max:
                            other_max = rows[i][-1]
                new_scope = max(other_max, new_row_max)
                
                # Accept
                if new_scope < current_scope or random.random() < pow(2.718281828, (current_scope - new_scope) / initial_temp):
                    # Update used_diffs
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    
                    rows[row_idx][pos] = new_val
                    # Update row_diffs[row_idx]
                    row_diffs[row_idx] = calc_row_diffs(rows[row_idx])
                    current_scope = new_scope
                    accepted = True
                    break
            
            initial_temp *= cooling_rate
            steps += 1

    # --- Final Polish ---
    # Try to reduce the last element of the max row iteratively
    t_polish_end = deadline - 0.1
    while time.time() < t_polish_end:
        max_row_idx = -1
        max_val = -1
        for i, r in enumerate(rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_row_idx = i
        
        if max_row_idx == -1:
            break
            
        row = rows[max_row_idx]
        last_pos = k
        old_val = row[last_pos]
        
        # Try to reduce old_val by 1, 2, ...
        found_reduction = False
        for new_val in range(old_val - 1, row[last_pos - 1], -1):
            # Check validity
            old_diffs = set()
            for i in range(last_pos):
                old_diffs.add(old_val - row[i])
            
            new_diffs = set()
            valid = True
            for i in range(last_pos):
                d = new_val - row[i]
                if d in used_diffs and d not in old_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Apply
                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                
                row[last_pos] = new_val
                row_diffs[max_row_idx] = calc_row_diffs(row)
                current_scope = get_scope(rows)
                found_reduction = True
                break
        
        if not found_reduction:
            # Try rebuilding this row with scope current_scope - 1
            forbidden = set()
            for i in range(n_rows):
                if i != max_row_idx:
                    forbidden |= row_diffs[i]
            
            new_row = backtrack_row(forbidden, k, current_scope - 1)
            if new_row is not None:
                new_row_diffs = calc_row_diffs(new_row)
                # It's valid by construction
                used_diffs.difference_update(row_diffs[max_row_idx])
                used_diffs.update(new_row_diffs)
                
                rows[max_row_idx] = new_row
                row_diffs[max_row_idx] = new_row_diffs
                current_scope = get_scope(rows)
            else:
                break # Stuck

    return rows

