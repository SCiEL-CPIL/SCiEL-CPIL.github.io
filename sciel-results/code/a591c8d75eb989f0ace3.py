import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry (simple counting bound)
    # The differences must be distinct integers from 1 to M.
    # So M >= total_diffs.
    lower_bound = total_diffs
    
    # Helper to compute the cost (energy) of a configuration
    # Energy = w1 * (max_entry)^2 + w2 * (num_conflicts)
    # We want to minimize this.
    # If num_conflicts == 0, the solution is valid, and we minimize max_entry.
    # If num_conflicts > 0, we pay a penalty to drive towards validity.
    # By allowing the search to move through invalid states, we can escape
    # local optima where a valid solution with a slightly larger max_entry
    # is trapped by a "wall" of collisions.
    
    def compute_energy(rows):
        used_diffs = {}
        max_entry = 0
        for row in rows:
            if row:
                m = max(row)
                if m > max_entry:
                    max_entry = m
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        used_diffs[d] += 1
                    else:
                        used_diffs[d] = 1
        
        num_conflicts = 0
        for count in used_diffs.values():
            if count > 1:
                num_conflicts += count - 1
        
        # Weight for max entry squared to strongly encourage small max
        # Weight for conflicts to ensure validity is reached
        # We tune these dynamically or fix them. 
        # Let's use a fixed penalty that is large enough to prioritize validity
        # once max_entry is close to the lower bound, but small enough to allow
        # temporary invalid states.
        
        # Actually, a better strategy:
        # If we are far from the lower bound, prioritize reducing max_entry.
        # If we are close, prioritize reducing conflicts.
        # But a simple composite energy works well in SA.
        
        # Let's use: Energy = max_entry + 100 * num_conflicts
        # Wait, if max_entry is 1000 and conflicts is 1, energy is 1100.
        # If max_entry is 1010 and conflicts is 0, energy is 1010.
        # This correctly prefers the valid solution.
        # But if max_entry is 1000 and conflicts is 10, energy is 2000.
        # This strongly penalizes conflicts.
        
        # To allow navigating through invalid states to find a smaller max_entry,
        # we need the penalty for conflicts to be "soft" enough that a move
        # increasing conflicts but significantly reducing max_entry is accepted
        # with some probability.
        
        # Let's use: Energy = max_entry + 50 * num_conflicts
        # This allows a trade-off.
        
        return max_entry + 50.0 * num_conflicts

    def generate_random_rows(max_val):
        rows = []
        for _ in range(n_rows):
            if max_val < k:
                row = list(range(k+1))
            else:
                vals = random.sample(range(1, max_val + 1), k)
                vals.sort()
                row = [0] + vals
            rows.append(row)
        return rows

    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def get_used_diffs(rows, exclude_idx=None):
        used = set()
        for i, row in enumerate(rows):
            if i == exclude_idx:
                continue
            for a in range(len(row)):
                for b in range(a+1, len(row)):
                    used.add(row[b] - row[a])
        return used

    def repair_row(rows, idx, max_limit):
        """
        Reconstruct row idx to minimize its max value while avoiding
        differences used by other rows.
        Returns the new row.
        """
        forbidden = get_used_diffs(rows, exclude_idx=idx)
        
        # Greedy construction: pick smallest available x
        row = [0]
        current_max = 0
        # We want to fit within max_limit
        # Try to construct a row with max <= max_limit
        
        # Strategy: Backtracking or simple greedy?
        # Simple greedy: pick smallest x > last such that all diffs (x - a for a in row) are not in forbidden.
        # This might get stuck if we pick too small a number early, blocking later numbers.
        # But for local search, a "best effort" greedy is okay.
        
        # Let's try a slightly smarter greedy: 
        # At each step, consider candidates from last+1 to max_limit.
        # Pick the one that leaves the most "flexibility"? 
        # Or just pick the smallest valid one.
        
        # Actually, picking the smallest valid one is standard for these "packing" problems 
        # to keep the range small.
        
        while len(row) < k + 1:
            last = row[-1]
            found = False
            for x in range(last + 1, max_limit + 1):
                # Check if x - a is in forbidden for all a in row
                valid = True
                for a in row:
                    if x - a in forbidden:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    # Update forbidden? No, we are constructing this row, 
                    # the diffs within this row are fine as long as they don't collide with OTHER rows.
                    # But wait, if we add x, the new diffs are x-a. 
                    # We don't need to add them to 'forbidden' because 'forbidden' only contains
                    # diffs from OTHER rows. 
                    # However, we must ensure that the NEW diffs don't collide with each other?
                    # No, within a row, diffs are automatically distinct if the row is strictly increasing?
                    # Wait. If row is [0, 1, 3], diffs are 1, 3, 2. All distinct.
                    # If row is [0, 2, 4], diffs are 2, 4, 2. Collision!
                    # So we MUST check that the new diffs (x - a for a in row) are distinct from each other too.
                    # Actually, if we pick x, the new diffs are x - row[0], x - row[1], ..., x - row[-1].
                    # These are distinct because row is strictly increasing.
                    # But do they collide with existing diffs in the row?
                    # Existing diffs are row[j] - row[i].
                    # New diffs are x - row[i].
                    # We need x - row[i] != row[j] - row[i'] for all valid i, j, i'.
                    # Specifically, we need to ensure that the set of new diffs is disjoint from the set of existing diffs in the row.
                    
                    # Let's re-evaluate validity including internal row collisions.
                    valid_internal = True
                    existing_diffs = set()
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            existing_diffs.add(row[j] - row[i])
                    
                    new_diffs = set()
                    for a in row:
                        new_diffs.add(x - a)
                    
                    if new_diffs & existing_diffs:
                        valid_internal = False
                        
                    if valid and valid_internal:
                        row.append(x)
                        found = True
                        break
            
            if not found:
                # Couldn't fit in max_limit
                # Return a "best effort" row or the previous one?
                # If we can't fit, we might need to relax max_limit or accept a larger row.
                # For now, return the row as is (incomplete) or a random one?
                # To maintain validity of the structure, let's just return the previous row (no change)
                # or a row that is valid internally but might collide externally.
                # Actually, if we can't find a valid x, we should probably stop and return the old row.
                return rows[idx][:]
                
        return row

    # Initial baseline
    baseline_rows = pure_greedy()
    best_rows = [row[:] for row in baseline_rows]
    best_max = max(max(row) for row in best_rows)
    
    # SA Parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.9995
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize SA
        # Start with a random configuration within a range
        # If we have a good best_max, we can start near it
        init_max = best_max + 5
        current_rows = generate_random_rows(init_max)
        
        current_energy = compute_energy(current_rows)
        
        # Check if initial is better
        curr_max = max(max(row) for row in current_rows)
        if curr_max < best_max:
            # Verify validity
            used = set()
            valid = True
            for row in current_rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in used:
                            valid = False
                            break
                        used.add(d)
                if not valid: break
            if valid:
                best_max = curr_max
                best_rows = [row[:] for row in current_rows]

        T = T_start
        
        # Hybrid loop: Alternate between SA steps and Row Repairs
        step_count = 0
        while T > T_end and time.time() < deadline:
            step_count += 1
            
            # Every 100 steps, try a row repair
            if step_count % 100 == 0:
                r_idx = random.randint(0, n_rows - 1)
                # Try to repair row r_idx to be smaller
                # Limit is current max of that row, or best_max
                limit = min(current_rows[r_idx][-1], best_max)
                new_row = repair_row(current_rows, r_idx, limit)
                
                if new_row != current_rows[r_idx]:
                    old_energy = current_energy
                    current_rows[r_idx] = new_row
                    current_energy = compute_energy(current_rows)
                    
                    # Check validity
                    used = set()
                    valid = True
                    for row in current_rows:
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                d = row[j] - row[i]
                                if d in used:
                                    valid = False
                                    break
                                used.add(d)
                        if not valid: break
                    
                    if valid:
                        curr_max = max(max(row) for row in current_rows)
                        if curr_max < best_max:
                            best_max = curr_max
                            best_rows = [row[:] for row in current_rows]
            else:
                # Standard SA step
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, k)
                
                min_val = current_rows[r_idx][c_idx - 1] + 1
                max_val = best_max + 10 # Allow exploration slightly above current best
                
                if c_idx < k:
                    max_val = min(max_val, current_rows[r_idx][c_idx + 1] - 1)
                
                if min_val > max_val:
                    T *= cooling_rate
                    continue
                    
                new_val = random.randint(min_val, max_val)
                
                if new_val == current_rows[r_idx][c_idx]:
                    T *= cooling_rate
                    continue
                    
                old_val = current_rows[r_idx][c_idx]
                current_rows[r_idx][c_idx] = new_val
                
                new_energy = compute_energy(current_rows)
                delta_energy = new_energy - current_energy
                
                # Acceptance probability
                if delta_energy < 0 or (T > 0 and random.random() < math.exp(-delta_energy / T)):
                    current_energy = new_energy
                    
                    # Check if we found a valid solution better than best
                    # We only need to check validity if the max_entry is potentially better
                    # or if we suspect we are close to valid.
                    # To save time, we only check validity if the current max is <= best_max
                    # or if the energy is very low (implying few conflicts).
                    
                    curr_max = max(max(row) for row in current_rows)
                    
                    # Heuristic: If max is already smaller than best, check validity.
                    # If max is equal to best, check validity.
                    # If max is larger, skip validity check unless energy is very low.
                    
                    should_check = (curr_max <= best_max) or (new_energy < best_max + 10)
                    
                    if should_check:
                        used = set()
                        valid = True
                        for row in current_rows:
                            for i in range(len(row)):
                                for j in range(i+1, len(row)):
                                    d = row[j] - row[i]
                                    if d in used:
                                        valid = False
                                        break
                                    used.add(d)
                            if not valid: break
                        
                        if valid and curr_max < best_max:
                            best_max = curr_max
                            best_rows = [row[:] for row in current_rows]
                else:
                    # Reject
                    current_rows[r_idx][c_idx] = old_val
                    
            T *= cooling_rate

    return best_rows
