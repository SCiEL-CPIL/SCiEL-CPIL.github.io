import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    total_diffs_per_row = k * (k + 1) // 2
    counting_lb = n_rows * total_diffs_per_row

    def row_diffs(row):
        s = set()
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ri)
        return s

    def is_valid(rows):
        all_d = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True

    def greedy_row(forbidden, max_val_hint):
        """Greedily construct a row avoiding forbidden diffs, staying under max_val_hint."""
        row = [0]
        local = set()
        for _ in range(k):
            base = row[-1]
            found = False
            # Scan upwards from base+1
            for v in range(base + 1, max_val_hint + 1):
                ok = True
                for a in row:
                    d = v - a
                    if d in forbidden or d in local:
                        ok = False
                        break
                if ok:
                    row.append(v)
                    for a in row[:-1]:
                        local.add(row[-1] - a)
                    found = True
                    break
            if not found:
                return None
        return row

    def initial_construct(max_hint):
        rows = []
        used = set()
        for i in range(n_rows):
            r = greedy_row(used, max_hint)
            if r is None:
                return None
            rows.append(r)
            used |= row_diffs(r)
        return rows

    # 1. Initial Construction
    # Start with a reasonable hint. The counting bound is a lower bound on the max value.
    # For (7,5), lb = 7 * 15 = 105. Known best is 112.
    hint = max(counting_lb + 5, 110)
    init_rows = initial_construct(hint)
    
    if init_rows is None:
        # Fallback: very loose greedy
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        init_rows = rows

    def score_of(rows):
        return max(max(r) for r in rows)

    best_rows = [r[:] for r in init_rows]
    best_score = score_of(best_rows)
    cur_rows = [r[:] for r in init_rows]
    cur_row_diffs = [row_diffs(r) for r in cur_rows]
    cur_score = score_of(cur_rows)

    # 2. Backtracking Row Solver
    # Tries to find a row that fits in the 'forbidden' set and has max <= max_allowed.
    def backtrack_find_row(forbidden, max_allowed):
        """
        Finds a valid row of length k+1 starting at 0, strictly increasing,
        with all differences disjoint from 'forbidden', and max entry <= max_allowed.
        Returns the row or None.
        """
        if time.time() > deadline:
            return None
            
        row = [0]
        local_diffs = set()
        
        def dfs(pos, current_max):
            if pos == k:
                # We have k elements after 0, so total k+1 elements.
                # Check if max entry is within limit
                if row[-1] <= max_allowed:
                    return True
                return False
            
            base = row[-1]
            # Heuristic: try smaller values first to keep max low, 
            # but also consider that we need to fit k-pos more numbers.
            # The next number must be at least base + 1.
            # The maximum possible value for the last number is max_allowed.
            # So the next number v must satisfy:
            # v + (k - pos - 1) <= max_allowed  (rough estimate, assuming steps of 1)
            # Actually, just bound the scan.
            
            min_v = base + 1
            # We need to leave room for (k - pos - 1) more numbers, each at least +1.
            # So v + (k - pos - 1) <= max_allowed is a necessary condition for feasibility?
            # Not strictly, because gaps can be larger, but if v is too large, we might not fit?
            # No, if v is large, the remaining numbers are even larger. 
            # Wait, we want to MINIMIZE the max. So we prefer small v.
            # But we must ensure we CAN complete the row.
            # If we pick v, the next min is v+1. The last min is v + (k-pos-1).
            # So we require v + (k - pos - 1) <= max_allowed.
            max_v = max_allowed - (k - pos - 1)
            
            if min_v > max_v:
                return False

            # Iterate candidates
            # To improve search, we can randomize the order slightly or stick to greedy.
            # Greedy (smallest first) is usually good for minimizing max.
            for v in range(min_v, max_v + 1):
                if time.time() > deadline:
                    return False
                
                # Check conflicts
                ok = True
                for a in row:
                    d = v - a
                    if d in forbidden or d in local_diffs:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Add v
                row.append(v)
                new_diffs = set()
                for a in row[:-1]:
                    d = v - a
                    new_diffs.add(d)
                local_diffs.update(new_diffs)
                
                if dfs(pos + 1, v):
                    return True
                
                # Backtrack
                row.pop()
                for d in new_diffs:
                    local_diffs.discard(d)
            
            return False

        if dfs(0, 0):
            return row[:]
        return None

    # 3. Main Search Loop: Hybrid SA + Backtracking Repair
    # We use SA to explore the space, but the "move" is a backtracking repair of one row.
    # This is more powerful than greedy replacement.
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    no_improve_count = 0
    max_no_improve = 500  # Restart if stuck

    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        # Pick a row to replace
        idx = random.randrange(n_rows)
        
        # Calculate forbidden diffs (all diffs from other rows)
        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= cur_row_diffs[i]
        
        # Determine the target max value for the new row.
        # We want to improve the global score.
        # Global score is max of all row maxes.
        # If we replace row idx, the new global score will be max(max_of_other_rows, max_of_new_row).
        # Let M_others = max(max(r) for r in cur_rows if r != cur_rows[idx])
        M_others = 0
        for i in range(n_rows):
            if i != idx:
                if cur_rows[i][-1] > M_others:
                    M_others = cur_rows[i][-1]
        
        # Current row max
        cur_row_max = cur_rows[idx][-1]
        
        # We want the new row to have max <= M_others if possible, to not increase global score.
        # Or even better, if cur_row_max was the global max, we want new_max < cur_row_max.
        
        # Strategy:
        # 1. Try to find a row with max <= M_others. This keeps global score <= M_others.
        # 2. If that fails, try to find a row with max < cur_row_max (if cur_row_max > M_others).
        # 3. If that fails, accept any valid row (might increase score, but SA can handle it).
        
        target_max = M_others
        
        # If the current row is NOT the unique max, M_others is the global max.
        # We try to keep it below or equal to M_others.
        
        new_row = backtrack_find_row(forbidden, target_max)
        
        # If we couldn't find one within M_others, try a slightly looser bound if we are in "exploration" mode
        # or if current score is already low.
        if new_row is None and T > min_T:
            # Allow a small increase in score to escape local minima
            # Only allow increase if we haven't improved in a while
            if no_improve_count > 100:
                # Try to find a row that is at least as good as current, or slightly worse
                # But we must ensure we don't blow up the score.
                # Let's try max = cur_row_max
                new_row = backtrack_find_row(forbidden, cur_row_max)
        
        if new_row is None:
            # No valid row found even with loose constraints? 
            # This might happen if the forbidden set is too dense.
            # In this case, we just cool down.
            T *= cooling
            no_improve_count += 1
            continue
            
        new_row_d = row_diffs(new_row)
        
        # Verify validity (should be guaranteed by backtrack, but safe check)
        if new_row_d & forbidden:
            T *= cooling
            continue
            
        new_rows = cur_rows[:]
        new_rows[idx] = new_row
        new_score = score_of(new_rows)
        
        delta = new_score - cur_score
        
        accept = False
        if delta < 0:
            accept = True
            no_improve_count = 0
        elif delta == 0:
            # Accept equal score moves with some probability to diversify
            if random.random() < 0.05:
                accept = True
                no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Accept worse moves based on SA
            prob = math.exp(-delta / T)
            if random.random() < prob:
                accept = True
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        if accept:
            cur_rows[idx] = new_row
            cur_row_diffs[idx] = new_row_d
            cur_score = new_score
            
            if cur_score < best_score:
                best_score = cur_score
                best_rows = [r[:] for r in cur_rows]
                no_improve_count = 0
                # Optional: restart with higher T if we found a new best?
                # T = max(T, 5.0) 
                
        T *= cooling
        if T < min_T:
            T = 10.0 # Restart temperature
            no_improve_count = 0

    # Final Validation
    if not is_valid(best_rows):
        # Fallback to a simple greedy construction if something went wrong
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return best_rows

# EVOLVE-BLOCK-END
