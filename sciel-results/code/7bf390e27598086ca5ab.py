import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 120.0
    
    # Edge case: if k is very small, handle directly or let general logic handle it
    # k is number of intervals, so k+1 elements.
    
    # 1. Greedy Initialization
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        break
                    x += 1
                row.append(x)
                used_diffs |= new_diffs
            rows.append(row)
        return rows

    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_scope = max(max(r) for r in current_rows)
    
    # Helper to calculate scope and conflicts
    def get_scope_conflicts(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            max_val = max(max_val, max(r))
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        return max_val, conflicts

    current_scope, current_conflicts = get_scope_conflicts(current_rows)
    current_cost = current_scope + 1000 * current_conflicts
    
    # Simulated Annealing
    # T starts high, decays
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    # Pre-calculate global used diffs for faster check?
    # It's easier to just recalculate or maintain a set.
    # Maintaining a set is better for speed.
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used.add(r[j] - r[i])
        return used

    used_diffs = get_used_diffs(current_rows)
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            break
            
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        # Pick a random element to change (not the first one, which is 0)
        # We can change any element from index 1 to k
        j_idx = random.randint(1, k)
        
        # Pick a delta
        # Bias towards smaller values if we are trying to reduce scope?
        # For now, symmetric random
        delta = random.randint(-10, 10)
        
        old_val = current_rows[r_idx][j_idx]
        new_val = old_val + delta
        
        # Check bounds
        # Must be > previous and < next
        if j_idx > 0:
            prev_val = current_rows[r_idx][j_idx - 1]
            if new_val <= prev_val:
                T *= cooling_rate
                continue
        if j_idx < k:
            next_val = current_rows[r_idx][j_idx + 1]
            if new_val >= next_val:
                T *= cooling_rate
                continue
        
        # Check if new_val is 0? No, strictly increasing from 0.
        if new_val <= 0:
            T *= cooling_rate
            continue
            
        # Calculate new differences for this row
        row = current_rows[r_idx]
        new_row = row[:]
        new_row[j_idx] = new_val
        
        # Old diffs involving j_idx
        old_diffs = set()
        for i in range(j_idx):
            old_diffs.add(old_val - row[i])
        for i in range(j_idx + 1, len(row)):
            old_diffs.add(row[i] - old_val)
        
        # New diffs involving j_idx
        new_diffs = set()
        for i in range(j_idx):
            new_diffs.add(new_val - new_row[i])
        for i in range(j_idx + 1, len(new_row)):
            new_diffs.add(new_row[i] - new_val)
        
        # Check conflicts
        # The global set of diffs should not contain any new_diffs
        # EXCEPT if they were in old_diffs (which we are removing)
        # Actually, simpler:
        # 1. Remove old_diffs from used_diffs
        # 2. Check if any new_diff is in used_diffs
        # 3. If not, add new_diffs to used_diffs
        # 4. If conflict, revert
        
        # To avoid modifying used_diffs and reverting on reject, we can just check:
        # Is there any d in new_diffs such that d is in used_diffs and d is NOT in old_diffs?
        # If yes, conflict.
        
        has_conflict = False
        for d in new_diffs:
            if d in used_diffs and d not in old_diffs:
                has_conflict = True
                break
        
        new_conflicts = 1 if has_conflict else 0
        # Note: This logic is slightly incomplete if there are multiple conflicts or if old_diffs had conflicts internally (they don't, as current is valid or we track it).
        # Actually, if current state has conflicts, this logic is tricky.
        # Let's assume we always maintain a state where we know the conflict count.
        
        # Let's refine:
        # current_conflicts is the number of duplicated diffs in the whole set.
        # If we change row r_idx, the diffs that change are old_diffs -> new_diffs.
        # We need to recalculate the conflict contribution of these diffs.
        
        # This is getting complex. Let's use a simpler but slightly slower check:
        # Just compute the new cost from scratch?
        # No, K is small.
        
        # Let's try a different move strategy: **Greedy Hill Climbing**
        # Only accept if cost decreases.
        # If stuck, restart with a perturbed greedy solution.
        
        # Let's stick to SA but with a robust cost evaluation.
        # Since N, K are small, we can afford to be a bit less efficient in conflict checking if it's simpler.
        
        # Let's calculate new_scope and new_conflicts for the whole set?
        # No, that's O(N K^2).
        
        # Let's use the incremental check properly.
        # We maintain `used_diffs` as a set of all diffs.
        # We also maintain `conflict_count`.
        
        # If has_conflict is False:
        #   We need to check if removing old_diffs and adding new_diffs changes conflict_count.
        #   This is hard to do incrementally without a frequency map.
        
        # Let's use a Frequency Map of differences.
        # `diff_freq[d]` = number of times difference d appears in the current solution.
        # Conflict count = sum(1 for d in diff_freq if diff_freq[d] > 1).
        
        # Let's rebuild the state with a frequency map.
        
    # Restart with Frequency Map approach for better SA
    
    def get_diff_freq(rows):
        freq = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    freq[d] = freq.get(d, 0) + 1
        return freq

    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_scope = max(max(r) for r in current_rows)
    
    freq = get_diff_freq(current_rows)
    current_scope = best_scope
    # Conflict count
    current_conflicts = sum(1 for d, c in freq.items() if c > 1)
    current_cost = current_scope + 1000 * current_conflicts
    
    T = 50.0
    T_min = 0.5
    cooling = 0.999
    
    while time.time() < deadline - 1.0:
        if T < T_min:
            break
            
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k)
        
        # Bias delta towards negative if this element is large?
        # Let's just use random
        delta = random.randint(-5, 5)
        
        old_val = current_rows[r_idx][j_idx]
        new_val = old_val + delta
        
        if new_val <= 0:
            T *= cooling
            continue
            
        # Check bounds
        if j_idx > 0 and new_val <= current_rows[r_idx][j_idx-1]:
            T *= cooling
            continue
        if j_idx < k and new_val >= current_rows[r_idx][j_idx+1]:
            T *= cooling
            continue
            
        row = current_rows[r_idx]
        
        # Old diffs
        old_diffs = []
        for i in range(j_idx):
            old_diffs.append(old_val - row[i])
        for i in range(j_idx+1, len(row)):
            old_diffs.append(row[i] - old_val)
            
        # New diffs
        new_diffs = []
        for i in range(j_idx):
            new_diffs.append(new_val - row[i])
        for i in range(j_idx+1, len(row)):
            new_diffs.append(row[i] - new_val)
            
        # Check validity of new diffs (must be positive, which is guaranteed by bounds)
        
        # Calculate change in conflict count
        # We remove old_diffs and add new_diffs
        # We need to check if any new_diff is already in freq (after removing old_diffs)
        
        # To do this efficiently:
        # 1. Temporarily decrement freq for old_diffs
        # 2. Check if any new_diff has freq > 0
        # 3. If yes, conflict.
        # 4. If no, increment freq for new_diffs
        # 5. Calculate delta_conflicts
        
        # Let's just do it.
        
        # Decrement old
        for d in old_diffs:
            freq[d] -= 1
            if freq[d] == 0:
                del freq[d]
                
        # Check new
        new_conflicts = 0
        valid = True
        for d in new_diffs:
            if d in freq:
                # This diff already exists, so adding it creates a conflict
                # But wait, if freq[d] was 1, it becomes 2 (conflict).
                # If freq[d] was 2, it becomes 3 (still conflict, count doesn't change? No, the number of conflicting pairs increases, but we count "number of diffs with freq>1" or "number of conflicts"?
                # Let's count "number of differences that are duplicated".
                # If d was 1, now 2 -> +1 conflict.
                # If d was 2, now 3 -> 0 change in count of "duplicated diffs"?
                # Actually, minimizing scope is the goal. Conflicts just need to be 0.
                # So we just need to know if the new state has ANY conflicts.
                # If we are in a valid state (0 conflicts), and we add a diff that exists, we have 1 conflict.
                # If we are in an invalid state, we want to reduce conflicts.
                
                # Let's stick to: Cost = Scope + 1000 * (1 if any conflict else 0).
                # This is a hard constraint.
                # If we allow soft constraints, we need a better metric.
                
                # Let's use: Cost = Scope + 1000 * total_conflict_count.
                # total_conflict_count = sum(max(0, freq[d]-1) for d in freq).
                
                pass
                
        # This is getting complicated to track incrementally.
        # Given N, K are small, let's just recalculate cost for the candidate state.
        # It's O(N K^2) ~ 100 ops.
        # We can do 100,000 moves in 120s. That's plenty.
        
        # Candidate state
        cand_rows = [r[:] for r in current_rows]
        cand_rows[r_idx][j_idx] = new_val
        
        cand_scope, cand_conflicts = get_scope_conflicts(cand_rows)
        cand_cost = cand_scope + 1000 * cand_conflicts
        
        delta_cost = cand_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_rows = cand_rows
            current_scope = cand_scope
            current_conflicts = cand_conflicts
            current_cost = cand_cost
            
            if cand_conflicts == 0 and cand_scope < best_scope:
                best_scope = cand_scope
                best_rows = [r[:] for r in current_rows]
                
        T *= cooling
        
    return best_rows
