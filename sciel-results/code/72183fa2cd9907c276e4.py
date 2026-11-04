"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit (actually 150s limit, but keep safe)
    
    # 1. Generate a baseline using pure greedy
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
        return rows, max(max(r) for r in rows)

    baseline_rows, baseline_max = pure_greedy()
    best_rows = [r[:] for r in baseline_rows]
    best_max = baseline_max
    
    # If we already hit the theoretical lower bound, we can't do better
    lower_bound = n_rows * k * (k + 1) // 2
    if best_max == lower_bound:
        return best_rows

    # 2. Simulated Annealing
    # State: rows (list of lists), used (set of all diffs)
    # We need to maintain 'used' efficiently.
    # Instead of rebuilding 'used' from scratch (O(N*k^2)), we update it incrementally.
    
    current_rows = [r[:] for r in baseline_rows]
    current_used = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                current_used.add(r[j] - r[i])
    
    current_max = max(max(r) for r in current_rows)
    
    # SA Parameters
    T = 50.0  # Initial temperature
    cooling = 0.995
    min_T = 0.01
    
    rng = random.Random(42)
    
    def get_max(row):
        return row[-1] # Since rows are increasing, max is last element

    def calc_global_max(rows):
        return max(r[-1] for r in rows)

    def check_move_validity(row, idx, new_val, current_used, old_val):
        """
        Check if replacing row[idx] with new_val maintains validity.
        Returns True if valid, False otherwise.
        Assumes row is strictly increasing with new_val.
        """
        # The differences that change are those involving index 'idx'
        # Old diffs: {old_val - row[j] for j < idx} U {row[j] - old_val for j > idx}
        # New diffs: {new_val - row[j] for j < idx} U {row[j] - new_val for j > idx}
        
        # We need to ensure new_diffs do not conflict with (current_used - old_diffs)
        # And new_diffs are distinct (guaranteed by strict increase)
        
        # Optimization: 
        # 1. Compute new_diffs
        # 2. Compute old_diffs
        # 3. Check intersection of new_diffs with (current_used - old_diffs)
        
        # Since k is small (e.g., 7), this is fast.
        
        new_diffs = set()
        for j in range(idx):
            d = new_val - row[j]
            new_diffs.add(d)
        for j in range(idx + 1, len(row)):
            d = row[j] - new_val
            new_diffs.add(d)
        
        old_diffs = set()
        for j in range(idx):
            d = old_val - row[j]
            old_diffs.add(d)
        for j in range(idx + 1, len(row)):
            d = row[j] - old_val
            old_diffs.add(d)
            
        # We can't just do set difference on current_used because we don't want to modify it yet.
        # We check if any d in new_diffs is in current_used AND not in old_diffs.
        # If d is in old_diffs, it's being removed, so it's okay if it's in new_diffs?
        # No, if d is in old_diffs, it means it was used by this specific pair.
        # If we reuse the same difference value in the new configuration for a different pair,
        # it's only valid if that difference is NOT used by ANY OTHER pair in the entire set.
        # So:
        # Conflict if d in new_diffs AND d in current_used AND d not in old_diffs.
        # Wait, if d is in old_diffs, it is removed from the global pool.
        # So if d is in new_diffs, it's re-added.
        # The only conflict is if d is in new_diffs and d is in (current_used - old_diffs).
        
        # Efficient check:
        # candidates = new_diffs
        # forbidden = current_used - old_diffs
        # if candidates & forbidden: return False
        # else return True
        
        # To avoid creating the large set difference 'forbidden' every time,
        # we can iterate new_diffs and check:
        # for d in new_diffs:
        #    if d in current_used and d not in old_diffs:
        #        return False
        # return True
        
        # This is O(k^2) worst case per check, which is fine for k=7.
        
        for d in new_diffs:
            if d in current_used:
                if d not in old_diffs:
                    return False
        return True

    def apply_move(row, idx, new_val, current_used, old_val):
        # Update current_used
        # Remove old_diffs
        for j in range(idx):
            current_used.discard(old_val - row[j])
        for j in range(idx + 1, len(row)):
            current_used.discard(row[j] - old_val)
        
        # Add new_diffs
        for j in range(idx):
            current_used.add(new_val - row[j])
        for j in range(idx + 1, len(row)):
            current_used.add(row[j] - new_val)
            
        row[idx] = new_val

    def revert_move(row, idx, old_val, current_used, new_val):
        # Revert: remove new_diffs, add old_diffs
        for j in range(idx):
            current_used.discard(new_val - row[j])
        for j in range(idx + 1, len(row)):
            current_used.discard(row[j] - new_val)
        
        for j in range(idx):
            current_used.add(old_val - row[j])
        for j in range(idx + 1, len(row)):
            current_used.add(row[j] - old_val)
            
        row[idx] = old_val

    # SA Loop
    steps = 0
    while time.time() < deadline and T > min_T:
        # Pick a random row
        r_idx = rng.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a random index to change (not 0, since it must be 0)
        # We can change any index from 1 to k
        i_idx = rng.randint(1, k)
        
        old_val = row[i_idx]
        
        # Determine range for new_val
        # Must be > row[i_idx-1] and < row[i_idx+1] (if i_idx < k)
        # To encourage reduction of max, bias towards smaller values if possible?
        # Or just random in the valid range.
        
        lower_bound_val = row[i_idx - 1] + 1
        upper_bound_val = row[i_idx + 1] - 1 if i_idx < k else old_val + 20 # If last, allow some expansion? No, we want to minimize.
        # Actually, if we are trying to minimize max, we should probably restrict the search space.
        # But SA needs to explore.
        # Let's allow a window around the current value.
        # If i_idx == k (last element), we definitely want to try smaller values.
        
        if i_idx == k:
            # Try to shrink
            # Sample from [lower_bound_val, old_val]
            # With some probability, try a value slightly larger to escape local minima?
            # Let's stick to [lower_bound_val, old_val] for the last element to drive max down.
            # If stuck, we might need to expand temporarily.
            # Let's sample from [lower_bound_val, old_val + 5]
            max_search = old_val + 5
            if lower_bound_val > max_search:
                continue
            new_val = rng.randint(lower_bound_val, max_search)
        else:
            # Middle elements: sample from a window around old_val
            # [lower_bound_val, upper_bound_val]
            # To keep it local, maybe [old_val - 10, old_val + 10] intersected with bounds?
            # But we must respect strict increase.
            # Let's just pick a random valid integer in the full range.
            # Range size can be large if gaps are large.
            # To keep moves local, let's pick a random delta.
            delta = rng.randint(-10, 10)
            new_val = old_val + delta
            if new_val <= lower_bound_val or new_val > upper_bound_val:
                # Retry with a smaller delta or skip
                # Try a smaller range
                if upper_bound_val > lower_bound_val:
                    new_val = rng.randint(lower_bound_val, upper_bound_val)
                else:
                    continue
        
        # Check validity
        if check_move_validity(row, i_idx, new_val, current_used, old_val):
            # Calculate new max
            # Only row r_idx changes
            old_row_max = row[-1]
            # If i_idx == k, new_row_max = new_val
            # Else, new_row_max is still row[-1] (unless i_idx was the max, which is only if i_idx==k)
            # Wait, if i_idx < k, the max of the row is row[-1], which is unchanged.
            # So global max only changes if i_idx == k.
            
            if i_idx == k:
                new_global_max = max([r[-1] for r in current_rows if r is not row] + [new_val])
            else:
                new_global_max = current_max # Unchanged
            
            delta_cost = new_global_max - current_max
            
            # Acceptance probability
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                # Apply move
                apply_move(row, i_idx, new_val, current_used, old_val)
                current_max = new_global_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
                    if best_max == lower_bound:
                        break
            else:
                # Revert (no-op since we didn't apply)
                pass
        else:
            # Invalid move, ignore
            pass
            
        T *= cooling
        steps += 1
        
        # Periodic restart? Or just continue.
        # If we are stuck for a long time, we could restart from best_rows.
        # For now, standard SA.
        
    return best_rows
# EVOLVE-BLOCK-END
