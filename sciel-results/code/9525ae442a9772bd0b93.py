import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Helper to calculate cost
    def get_cost(rows):
        return max(max(row) for row in rows)
    
    # Helper to check if a candidate x is valid for a current row prefix
    # given the set of used diffs from OTHER rows
    def is_valid_x(x, current_row, used_diffs):
        for a in current_row:
            diff = x - a
            if diff in used_diffs:
                return False
        return True

    # Randomized Greedy Construction of a single row
    # used_diffs: set of diffs used by other rows
    # returns a row or None if failed
    def construct_row(used_diffs, k, randomness_param=0.0):
        row = [0]
        last = 0
        for _ in range(k):
            x = last + 1
            valid_candidates = []
            while True:
                if is_valid_x(x, row, used_diffs):
                    valid_candidates.append(x)
                    if len(valid_candidates) >= 5: # Look ahead a bit
                        break
                x += 1
                if x > 10000: # Fallback safety, though problem implies smaller bounds
                    break
            
            if not valid_candidates:
                return None
            
            # Selection strategy
            if randomness_param > 0 and len(valid_candidates) > 1:
                # Pick from the first few with some randomness
                # Weighted towards smaller ones
                weights = [1.0 / (i + 1) for i in range(len(valid_candidates))]
                total_w = sum(weights)
                r = random.random() * total_w
                acc = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen_idx = i
                        break
                x = valid_candidates[chosen_idx]
            else:
                x = valid_candidates[0]
                
            row.append(x)
            last = x
            
        return row

    # Initial Solution
    used_diffs = set()
    rows = []
    for i in range(n_rows):
        # Try to construct row
        row = None
        # Retry with different randomness if fails? 
        # For initialization, deterministic is fine, or low randomness
        attempts = 0
        while row is None and attempts < 50:
            row = construct_row(used_diffs, k, randomness_param=0.5)
            attempts += 1
            if row:
                # Add diffs to used
                for j in range(len(row)):
                    for l in range(j+1, len(row)):
                        used_diffs.add(row[l] - row[j])
                rows.append(row)
                break
        if row is None:
            # Fallback to deterministic if randomized fails
            row = construct_row(used_diffs, k, randomness_param=0.0)
            if row:
                for j in range(len(row)):
                    for l in range(j+1, len(row)):
                        used_diffs.add(row[l] - row[j])
                rows.append(row)
            else:
                # Should not happen for reasonable inputs, but handle gracefully
                # Just return empty or best effort? Task implies valid output required.
                # If we can't construct, we return a dummy valid structure?
                # No, we must return a valid structure. 
                # If initialization fails, we try a more aggressive greedy.
                pass 

    if len(rows) < n_rows:
        # Fallback: standard greedy
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used_diffs:
                    row.append(x)
                    used_diffs |= new
            rows.append(row)

    best_rows = [r[:] for r in rows]
    best_cost = get_cost(best_rows)
    current_rows = [r[:] for r in rows]
    current_cost = best_cost
    
    # Rebuild used_diffs for current state
    def build_used(rows):
        s = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    s.add(r[j] - r[i])
        return s

    used_diffs = build_used(current_rows)
    
    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Pick a random row to rebuild
        idx = random.randint(0, n_rows - 1)
        
        # Remove its diffs
        old_row = current_rows[idx]
        diffs_to_remove = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                diffs_to_remove.add(old_row[j] - old_row[i])
        
        # We must be careful: if a diff is used by multiple rows, removing it
        # from the 'used' set is incorrect if another row also uses it.
        # However, the constraint is that ALL differences are distinct over ALL rows.
        # So no diff can appear in two different rows.
        # Thus, removing the diffs of row idx from the global set is safe 
        # ONLY IF we assume the global set contains *only* diffs from current rows.
        # Yes, that's the invariant.
        
        for d in diffs_to_remove:
            used_diffs.discard(d)
        
        # Try to rebuild row idx
        # Use randomness to explore
        new_row = construct_row(used_diffs, k, randomness_param=1.0)
        
        if new_row is None:
            # Revert
            used_diffs.update(diffs_to_remove)
            continue
        
        # Calculate new cost
        # Temporarily update state
        old_max = max(old_row)
        new_max = max(new_row)
        
        # We need to check if this move is valid. 
        # Since construct_row ensures no conflict with used_diffs (which excludes row idx),
        # and we removed row idx's old diffs, the new row is valid w.r.t others.
        
        new_rows = current_rows[:]
        new_rows[idx] = new_row
        new_cost = get_cost(new_rows)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_rows = new_rows
            current_cost = new_cost
            
            # Update used_diffs
            # Add new diffs
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    used_diffs.add(new_row[j] - new_row[i])
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject, revert used_diffs
            used_diffs.update(diffs_to_remove)
        
        T *= cooling_rate
        
    return best_rows

# EVOLVE-BLOCK-END
