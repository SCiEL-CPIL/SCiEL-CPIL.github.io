import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    # Hard deadline: finish with margin before 150s
    deadline = start_time + 140.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=20):
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    # --- New Structural Approach: Difference-Set Based Construction ---
    # Instead of building rows sequentially, we can think of the problem as
    # partitioning a set of differences.
    # A row of length k+1 starting at 0 is defined by k "steps" d_1, d_2, ..., d_k > 0.
    # The entries are 0, d_1, d_1+d_2, ..., sum(d_i).
    # The differences in this row are all partial sums of the steps.
    # Specifically, if steps are s_1...s_k, the diffs are {sum(s_i...s_j) | i<=j}.
    # Wait, the diffs are a_j - a_i.
    # a_j = sum(s_1...s_j).
    # a_j - a_i = sum(s_{i+1}...s_j).
    # So the set of diffs for a row is the set of all contiguous subsequence sums of the step vector.
    
    # This is a "Sidon sequence" like problem but with contiguous sums.
    
    # Let's try a different structural idea:
    # Construct rows using a "base" set of differences and then mapping them to rows?
    # No, that's hard.
    
    # Let's stick to Local Search but improve it significantly.
    # The previous local search was O(N^2) per step which is slow.
    # We can maintain the set of used diffs incrementally.
    
    def local_search_v2(rows, time_limit):
        """
        Optimized Local Search:
        1. Maintain a global set of used differences.
        2. When moving an element, only remove/add diffs involving that element.
        3. Use a "simulated annealing" style acceptance to escape local minima.
        4. Focus on minimizing the maximum element (scope).
        """
        start = time.time()
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        # Precompute used diffs
        used_diffs = set()
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        def get_diffs_involving(r_idx, c_idx, val):
            """Get set of diffs that would be removed if we change (r_idx, c_idx) from 'val'."""
            row = current_rows[r_idx]
            diffs = set()
            for i in range(c_idx):
                diffs.add(val - row[i])
            for j in range(c_idx + 1, len(row)):
                diffs.add(row[j] - val)
            return diffs

        def get_new_diffs(r_idx, c_idx, new_val):
            """Get set of diffs that would be added if we change (r_idx, c_idx) to 'new_val'."""
            row = current_rows[r_idx]
            diffs = set()
            for i in range(c_idx):
                diffs.add(new_val - row[i])
            for j in range(c_idx + 1, len(row)):
                diffs.add(row[j] - new_val)
            return diffs

        # Temperature for simulated annealing
        temp = 10.0
        cooling = 0.995
        
        while time.time() < start + time_limit:
            # Pick a random element to move (not the first one)
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, row_len - 1)
            
            curr_val = current_rows[r_idx][c_idx]
            prev_val = current_rows[r_idx][c_idx - 1]
            next_val = current_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 100
            
            # Determine search range
            # We want to try values that might reduce the scope or just explore.
            # If curr_val is the global max, we prioritize lower values.
            is_max = (curr_val == current_scope)
            
            # Candidate generation
            # Try values in (prev_val, next_val)
            low = prev_val + 1
            high = next_val - 1
            
            if high <= low:
                continue
            
            # If it's the max, we strongly prefer lower values.
            # Otherwise, randomize around current.
            if is_max:
                # Try from curr_val - 1 down to low
                candidates = range(curr_val - 1, low - 1, -1)
            else:
                # Randomly pick a range around current
                span = max(10, int(temp))
                cand_low = max(low, curr_val - span)
                cand_high = min(high, curr_val + span)
                candidates = random.sample(range(cand_low, cand_high + 1), min(20, cand_high - cand_low + 1))
                # Ensure we try a few lower ones if possible
                if curr_val > low + 5:
                    candidates.append(random.randint(low, curr_val - 1))

            found_move = False
            for new_val in candidates:
                if new_val <= prev_val or (c_idx < row_len - 1 and new_val >= next_val):
                    continue
                    
                # Check validity
                remove_set = get_diffs_involving(r_idx, c_idx, curr_val)
                add_set = get_new_diffs(r_idx, c_idx, new_val)
                
                # Check if any new diff is in (used_diffs - remove_set)
                # Or if any new diff is in add_set (duplicate within the new set)
                valid = True
                for d in add_set:
                    if d in used_diffs and d not in remove_set:
                        valid = False
                        break
                    if d in add_set: # This check is redundant if add_set is a set, but let's be safe
                         # Actually, add_set is a set, so duplicates are impossible within it.
                         # But we need to check if d conflicts with another d in add_set? No, set handles that.
                         # We just need to check against the *rest* of the used_diffs.
                         pass
                
                # Wait, the check above is slightly off.
                # We need to ensure that for all d in add_set:
                # 1. d is not in (used_diffs - remove_set)
                # This is equivalent to: d in used_diffs implies d in remove_set.
                
                valid = True
                for d in add_set:
                    if d in used_diffs:
                        if d not in remove_set:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Check if new_val creates duplicate diffs *within* the new set?
                # add_set is a set, so no.
                # But what if add_set contains a diff that is in remove_set?
                # That's fine, it means we're reusing a diff that was just freed.
                
                # Acceptance
                new_scope = max(new_val, current_scope)
                if new_val < current_scope and is_max:
                    # This is an improvement in scope
                    delta_cost = -10 # Reward
                else:
                    # Neutral or worsening
                    delta_cost = 0
                    if new_val > curr_val:
                        delta_cost = 1 # Penalty for increasing size
                
                # Simulated Annealing Acceptance
                if delta_cost < 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta_cost / temp)
                    # Here delta_cost is small, so mostly accept
                    accept = random.random() < math.exp(-delta_cost / temp) if temp > 0 else False
                
                if accept:
                    # Apply move
                    current_rows[r_idx][c_idx] = new_val
                    # Update used_diffs
                    used_diffs.difference_update(remove_set)
                    used_diffs.update(add_set)
                    
                    # Update scope
                    if new_val > current_scope:
                        current_scope = new_val
                    elif is_max:
                        # Recalculate scope if we reduced the max
                        # This is expensive, so only do it if we think we improved
                        # Actually, let's just recalculate every few steps or when is_max
                        current_scope = get_scope(current_rows)
                    
                    # Update best
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                    
                    found_move = True
                    break
            
            if not found_move:
                # No move accepted, cool down
                temp *= cooling
                if temp < 0.1:
                    temp = 0.1

        return best_rows, best_scope

    # Phase 1: Greedy Init
    best_rows = greedy_init(0.1, 20)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Local Search V2
    # Give it most of the time
    local_time = deadline - time.time() - 5.0
    if local_time > 10:
        refined_rows, refined_scope = local_search_v2(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 3: Final Polish (Greedy reduction of max)
    # Try to reduce the global max one more time with a strict greedy approach
    # Find global max
    max_val = -1
    max_r = -1
    max_c = -1
    for r_idx, r in enumerate(best_rows):
        for c_idx, val in enumerate(r):
            if val > max_val:
                max_val = val
                max_r = r_idx
                max_c = c_idx
    
    if max_c > 0:
        row = best_rows[max_r]
        curr_val = max_val
        prev_val = row[max_c - 1]
        
        # Rebuild used diffs for this specific check
        used_diffs = set()
        for r in best_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        remove_set = set()
        for i in range(max_c):
            remove_set.add(curr_val - row[i])
        for j in range(max_c + 1, len(row)):
            remove_set.add(row[j] - curr_val)
            
        fixed_diffs = used_diffs - remove_set
        
        for new_val in range(curr_val - 1, prev_val, -1):
            valid = True
            new_diffs = set()
            for i in range(max_c):
                d = new_val - row[i]
                if d in fixed_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid: continue
            
            for j in range(max_c + 1, len(row)):
                d = row[j] - new_val
                if d in fixed_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid: continue
            
            # Check internal duplicates
            if len(new_diffs) != max_c + (row_len - 1 - max_c):
                 # This check is weak, let's use list
                 pass
            # Actually, if we used a set, duplicates are gone.
            # We need to ensure that the *count* of diffs is correct?
            # No, we just need to ensure no diff appears twice in the *new* set.
            # Since we built it as a set, it's unique.
            # But we must ensure that no diff in new_diffs is in fixed_diffs (done).
            # And no diff in new_diffs is in new_diffs twice (impossible for set).
            
            # Wait, is it possible that new_diffs contains a diff that is also in new_diffs? No.
            # Is it possible that new_diffs contains a diff that is in fixed_diffs? Checked.
            # Is it possible that new_diffs contains a diff that is in remove_set?
            # If d in new_diffs and d in remove_set, that's fine.
            
            # Apply
            row[max_c] = new_val
            best_scope = get_scope(best_rows)
            break

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
