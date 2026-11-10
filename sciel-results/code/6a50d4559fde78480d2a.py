import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s)
    
    # Problem parameters
    # n_rows rows, each with k+1 elements (indices 0..k)
    # Element 0 is always 0.
    # Total differences per row = (k+1)*k/2
    # Total differences overall = n_rows * (k+1)*k/2
    # These differences must be distinct.
    # Minimize the maximum element (scope).
    
    # Precompute indices for differences
    # For a row of length m = k+1, differences are row[j] - row[i] for 0 <= i < j < m
    # There are m*(m-1)/2 differences.
    
    m = k + 1
    num_diffs_per_row = m * (m - 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Helper to compute differences of a row
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    # Helper to check validity and get scope
    def evaluate(rows):
        used_diffs = set()
        scope = 0
        valid = True
        for row in rows:
            if max(row) > scope:
                scope = max(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        valid = False
                        # Don't break immediately, we want to count conflicts for penalty maybe?
                        # But for now, just mark invalid.
                    used_diffs.add(d)
        return valid, scope

    # Greedy construction with randomization
    # We build rows one by one.
    # For each row, we choose elements x_1, ..., x_k such that 0 < x_1 < ... < x_k.
    # The differences x_j - x_i must not conflict with previous rows' diffs.
    # Also, within the row, differences must be distinct (Sidon property for the row itself? No, just distinct differences).
    # Actually, the condition is "all differences between two entries of the same row are distinct over all rows".
    # This implies:
    # 1. Within a row, all pair differences are distinct.
    # 2. Across rows, no difference value is repeated.
    
    def greedy_random_init(max_scope=200):
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            # We need to pick k more numbers.
            # To ensure distinct diffs within the row, we need a Sidon-like set (B_2 sequence) property locally?
            # Actually, if we pick numbers greedily ensuring new diffs don't collide with existing diffs in the row,
            # we maintain the property.
            
            # Randomized greedy:
            # At each step, pick the next element x > last.
            # Candidates: x such that {x - a for a in row} is disjoint from used_diffs AND disjoint from existing row diffs.
            # To speed up, we can limit the search range.
            
            last = 0
            for step in range(k):
                found = False
                # Search for next element
                # Range: from last+1 to last + some_gap
                # Let's try a window
                max_search = last + 50
                candidates = []
                for x in range(last + 1, max_search + 1):
                    new_diffs = {x - a for a in row}
                    # Check against global used_diffs
                    if any(d in used_diffs for d in new_diffs):
                        continue
                    # Check against internal row diffs
                    # We can maintain a set of internal diffs
                    # But for small k, checking is cheap.
                    # Let's check if new_diffs has any duplicates internally?
                    # x - a1 == x - a2 implies a1 == a2, which is false since row has distinct elements.
                    # So internal new diffs are always distinct from each other.
                    # We just need to ensure they don't collide with existing internal diffs.
                    # existing_internal_diffs is a set.
                    
                    # Optimization: Precompute internal diffs?
                    # For now, simple check.
                    internal_conflict = False
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in new_diffs:
                                internal_conflict = True
                                break
                        if internal_conflict:
                            break
                    if internal_conflict:
                        continue
                    
                    candidates.append(x)
                    if len(candidates) > 20:
                        break
                
                if not candidates:
                    # Failed to find a candidate in the window.
                    # This greedy might fail. Return None to trigger restart.
                    return None
                
                # Pick a candidate
                # Bias towards smaller values to keep scope low, but add randomness
                # Weighted random: weight = 1 / (x - last)
                weights = [1.0 / (c - last) for c in candidates]
                total_w = sum(weights)
                rand_val = random.random() * total_w
                cumulative = 0
                chosen = candidates[0]
                for i, c in enumerate(candidates):
                    cumulative += weights[i]
                    if rand_val <= cumulative:
                        chosen = c
                        break
                
                row.append(chosen)
                # Update used_diffs
                for a in row[:-1]:
                    used_diffs.add(chosen - a)
                last = chosen
            
            rows.append(row)
        
        valid, scope = evaluate(rows)
        if not valid:
            return None
        return rows

    # Better initialization: Use a known good structure or better greedy
    # Let's try multiple random restarts of greedy to find a good initial solution.
    
    best_rows = None
    best_scope = float('inf')
    
    # Try a few greedy inits
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        res = greedy_random_init()
        if res:
            valid, scope = evaluate(res)
            if valid and scope < best_scope:
                best_scope = scope
                best_rows = res
                if best_scope <= 112: # Theoretical lower bound
                    break
    
    if best_rows is None:
        # Fallback: deterministic greedy
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            last = 0
            for step in range(k):
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if any(d in used_diffs for d in new_diffs):
                        x += 1
                        continue
                    # Check internal
                    conflict = False
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            if row[j] - row[i] in new_diffs:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        x += 1
                        continue
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    last = x
                    break
            rows.append(row)
        best_rows = rows
        valid, best_scope = evaluate(best_rows)
    
    # Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Precompute difference pairs for fast evaluation?
    # For each row, the differences depend on the elements.
    # A move changes one element in one row.
    # This affects (k) differences in that row (differences involving that element).
    # Specifically, if we change row[r][i], the differences affected are:
    # row[r][i] - row[r][j] for j < i
    # row[r][j] - row[r][i] for j > i
    # Total k differences affected.
    
    # We need to track the set of all used differences to check for conflicts.
    # And we need to know if the new differences conflict with any other.
    
    # To make SA efficient:
    # 1. Maintain a global set of used_diffs.
    # 2. When moving an element, remove old diffs, add new diffs.
    # 3. Check for conflicts.
    # 4. If conflict, revert.
    
    # However, "conflict" means a difference appears twice.
    # We can maintain a count of each difference? Or just a set and check membership.
    # Since we remove old and add new, we can just check if new_diff is in (used_diffs - old_diffs).
    
    # Let's implement a fast update mechanism.
    
    def get_row_diffs_indices(row_idx, elem_idx):
        """
        Returns list of (i, j, diff) for the row, where i < j and the pair involves elem_idx.
        Actually, we just need the values of the differences to remove.
        """
        row = current_rows[row_idx]
        diffs = []
        for j in range(len(row)):
            if j == elem_idx:
                continue
            if j < elem_idx:
                diffs.append(row[elem_idx] - row[j])
            else:
                diffs.append(row[j] - row[elem_idx])
        return diffs

    # State:
    # current_rows
    # used_diffs_set: set of all current differences
    
    # Rebuild used_diffs_set from current_rows
    used_diffs_set = set()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs_set.add(row[j] - row[i])
    
    current_cost = current_scope
    
    T = 10.0
    alpha = 0.995
    min_T = 0.5
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 1:
        # Pick random row and element (1 to k, since 0 is fixed)
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)
        
        # Determine move range
        # Lower bound: current_rows[r_idx][i_idx-1] + 1
        # Upper bound: current_rows[r_idx][i_idx+1] - 1 (if i_idx < k)
        # Or allow expansion: current_rows[r_idx][i_idx] + 10
        # Or contraction: current_rows[r_idx][i_idx] - 10
        
        lower = current_rows[r_idx][i_idx - 1] + 1
        if i_idx < k:
            upper = current_rows[r_idx][i_idx + 1] - 1
        else:
            upper = current_rows[r_idx][i_idx] + 20
        
        # Also allow moving down if i_idx > 0
        # The move is changing the value at i_idx.
        # Let's pick a delta
        delta = random.randint(-10, 10)
        new_val = current_rows[r_idx][i_idx] + delta
        
        if new_val < lower or new_val > upper:
            continue
        
        # Calculate old diffs involving i_idx
        old_diffs = get_row_diffs_indices(r_idx, i_idx)
        
        # Calculate new diffs
        new_diffs = []
        for j in range(len(current_rows[r_idx])):
            if j == i_idx:
                continue
            if j < i_idx:
                new_diffs.append(new_val - current_rows[r_idx][j])
            else:
                new_diffs.append(current_rows[r_idx][j] - new_val)
        
        # Check validity of new_diffs
        # 1. Internal consistency: new_diffs must be distinct from each other?
        #    Yes, because the row elements are distinct.
        #    Wait, if we change one element, do the new diffs remain distinct from each other?
        #    new_diffs are {new_val - a_j for j < i} U {a_j - new_val for j > i}.
        #    Since a_j are distinct, new_val - a_j are distinct.
        #    a_j - new_val are distinct.
        #    Can new_val - a_j == a_l - new_val?
        #    2*new_val == a_j + a_l.
        #    This is possible. So we must check if any new_diff is repeated within new_diffs.
        #    Actually, if we just check against the global set, we need to be careful.
        
        # 2. Global consistency:
        #    Remove old_diffs from used_diffs_set.
        #    Check if any new_diff is in used_diffs_set.
        #    If so, conflict.
        #    Also check if any new_diff is in new_diffs (duplicates within the move).
        
        # Let's do a safe check:
        # Create a set of new_diffs to check for internal duplicates
        new_diffs_set = set(new_diffs)
        if len(new_diffs_set) != len(new_diffs):
            # Internal duplicate in the new diffs themselves
            continue
        
        # Remove old diffs from global set
        for d in old_diffs:
            used_diffs_set.discard(d)
        
        # Check conflicts
        conflict = False
        for d in new_diffs_set:
            if d in used_diffs_set:
                conflict = True
                break
        
        if conflict:
            # Revert: add old diffs back
            for d in old_diffs:
                used_diffs_set.add(d)
            continue
        
        # No conflict. Apply move.
        current_rows[r_idx][i_idx] = new_val
        for d in new_diffs_set:
            used_diffs_set.add(d)
        
        # Calculate new scope
        new_scope = max(max(r) for r in current_rows)
        
        # Cost function
        # We want to minimize scope.
        # If we have conflicts, we penalize heavily. But we already ensured no conflicts.
        # So cost is just scope.
        # However, we might want to encourage "tighter" packing?
        # For now, cost = scope.
        
        new_cost = new_scope
        delta_cost = new_cost - current_cost
        
        # Accept/Reject
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            current_scope = new_scope
            no_improve_steps = 0
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                if best_scope <= 112:
                    break
        else:
            # Revert
            current_rows[r_idx][i_idx] -= delta
            # Revert used_diffs_set
            for d in new_diffs_set:
                used_diffs_set.discard(d)
            for d in old_diffs:
                used_diffs_set.add(d)
            no_improve_steps += 1
        
        T *= alpha
        if T < min_T:
            T = 10.0
            no_improve_steps = 0 # Reset counter on reheat? Or keep?
            
        # Optional: Occasional row regeneration
        if random.random() < 0.01:
            # Pick a row, remove it, regenerate
            reg_idx = random.randint(0, n_rows - 1)
            # Remove diffs of reg_idx from used_diffs_set
            row_to_remove = current_rows[reg_idx]
            diffs_to_remove = get_diffs(row_to_remove)
            for d in diffs_to_remove:
                used_diffs_set.discard(d)
            
            # Regenerate row
            # Use greedy with remaining used_diffs
            new_row = [0]
            last = 0
            success = True
            for step in range(k):
                found = False
                for x in range(last + 1, last + 50):
                    new_diffs = {x - a for a in new_row}
                    if any(d in used_diffs_set for d in new_diffs):
                        continue
                    # Check internal
                    conflict = False
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            if new_row[j] - new_row[i] in new_diffs:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue
                    new_row.append(x)
                    for a in new_row[:-1]:
                        used_diffs_set.add(x - a)
                    last = x
                    found = True
                    break
                if not found:
                    success = False
                    break
            
            if success:
                current_rows[reg_idx] = new_row
                new_scope = max(max(r) for r in current_rows)
                new_cost = new_scope
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_cost = new_cost
                    current_scope = new_scope
                    no_improve_steps = 0
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= 112:
                            break
                else:
                    # Revert
                    # Remove new_row diffs
                    for d in get_diffs(new_row):
                        used_diffs_set.discard(d)
                    # Restore old row diffs
                    current_rows[reg_idx] = row_to_remove
                    for d in diffs_to_remove:
                        used_diffs_set.add(d)
                    no_improve_steps += 1

    return best_rows
