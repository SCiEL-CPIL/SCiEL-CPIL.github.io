import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 118.0
    best_max = float('inf')
    best_rows = None

    # --- Helper: Compute all pairwise differences for a row ---
    def get_diffs(row):
        diffs = []
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.append(row[j] - ri)
        return diffs

    # --- Helper: Greedy construction to get a valid baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # --- Helper: Greedy row rebuild with randomization ---
    # Attempts to build a row that fits within a max value and avoids used diffs
    def greedy_row_rebuild(used_set, max_val, row_len, rng, attempts=5):
        for attempt in range(attempts):
            row = [0]
            x = 0
            success = True
            for _ in range(row_len - 1):
                found = False
                # Randomize the start of the search range to escape local traps
                start_offset = 1
                if attempt > 0 and rng.random() < 0.5:
                    start_offset = rng.randint(1, 3)
                
                # We want to keep the row compact, so we search from low to high
                # but we must respect the previous element
                lower_bound = x + start_offset
                upper_bound = max_val
                
                # To add randomness, we might skip some small values
                # Or just try all values from lower_bound to upper_bound
                # For efficiency, we iterate and check
                for t in range(lower_bound, upper_bound + 1):
                    new_diffs = {t - a for a in row}
                    # Check internal uniqueness (implied by construction if we just add t > all previous? No, t-a can collide with other t-a')
                    # Actually, since row is sorted, t-a is strictly decreasing as a increases.
                    # So internal diffs are unique IF t-a values are distinct.
                    # t - row[0] > t - row[1] ... so they are distinct.
                    
                    # Check against used_set
                    if any(d in used_set for d in new_diffs):
                        continue
                    
                    row.append(t)
                    x = t
                    found = True
                    break
                
                if not found:
                    success = False
                    break
            
            if success:
                return row
        return None

    # --- Main Optimizer: Simulated Annealing ---
    def sa_optimize(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len
        
        # Precompute diffs and counts
        row_diffs = [get_diffs(r) for r in rows]
        
        # Global diff counts
        diff_counts = {}
        collision_count = 0
        for rd in row_diffs:
            for d in rd:
                if d in diff_counts:
                    collision_count += 1
                else:
                    diff_counts[d] = 0
                diff_counts[d] += 1
        
        # Cost function: Primary is collisions, Secondary is sum of entries (packing)
        # We want to minimize sum to tighten the rows.
        def compute_sum_val():
            return sum(sum(r) for r in rows)

        current_sum_val = compute_sum_val()
        # Weight for collisions must be large enough to dominate sum, 
        # but sum helps guide the search when collisions are 0.
        # If collisions > 0, we want to fix them.
        # If collisions == 0, we want to minimize sum.
        COLLISION_WEIGHT = 100000
        
        current_cost = collision_count * COLLISION_WEIGHT + current_sum_val
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        T = 100.0
        cooling = 0.999
        start_time = time.time()
        rng = random.Random(42)

        while time.time() - start_time < time_limit:
            # Choose move type
            # 70% Local Move, 30% Row Rebuild
            if rng.random() < 0.7:
                # --- Local Move: Change one element in a row ---
                r_idx = rng.randint(0, n - 1)
                i_idx = rng.randint(1, L - 1)
                
                row = rows[r_idx]
                old_val = row[i_idx]
                
                # Determine range
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < L - 1 else best_local_max + 10
                upper = min(upper, best_local_max + 20)
                
                if lower > upper:
                    continue
                    
                new_val = rng.randint(lower, upper)
                if new_val == old_val:
                    continue
                
                # Remove old diffs for this row
                old_diffs = row_diffs[r_idx]
                for d in old_diffs:
                    cnt = diff_counts[d]
                    if cnt == 2:
                        collision_count -= 1
                    diff_counts[d] = cnt - 1
                    if cnt - 1 == 0:
                        del diff_counts[d]
                
                # Apply change
                row[i_idx] = new_val
                new_diffs = get_diffs(row)
                row_diffs[r_idx] = new_diffs
                
                # Add new diffs
                for d in new_diffs:
                    if d in diff_counts:
                        if diff_counts[d] == 1:
                            collision_count += 1
                        diff_counts[d] += 1
                    else:
                        diff_counts[d] = 1
                
                new_sum_val = current_sum_val + (new_val - old_val)
                new_cost = collision_count * COLLISION_WEIGHT + new_sum_val
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_sum_val = new_sum_val
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_rows = [r[:] for r in rows]
                        best_local_max = max(r[-1] for r in best_local_rows)
                else:
                    delta = new_cost - current_cost
                    if delta == 0:
                        accept = rng.random() < 0.5
                    else:
                        # Standard SA acceptance
                        if delta < 500: # Only consider small jumps
                            p = math.exp(-delta / T)
                            accept = rng.random() < p
                        else:
                            accept = False
                    
                    if not accept:
                        # Revert
                        row[i_idx] = old_val
                        for d in new_diffs:
                            cnt = diff_counts[d]
                            if cnt == 2:
                                collision_count -= 1
                            diff_counts[d] = cnt - 1
                            if cnt - 1 == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            if d in diff_counts:
                                if diff_counts[d] == 1:
                                    collision_count += 1
                            diff_counts[d] += 1
                        row_diffs[r_idx] = old_diffs

            else:
                # --- Row Rebuild: Regenerate a whole row ---
                r_idx = rng.randint(0, n - 1)
                old_row = rows[r_idx][:]
                old_diffs = row_diffs[r_idx][:]
                
                # Remove old diffs from global counts
                for d in old_diffs:
                    cnt = diff_counts[d]
                    if cnt == 2:
                        collision_count -= 1
                    diff_counts[d] = cnt - 1
                    if cnt - 1 == 0:
                        del diff_counts[d]
                
                # Build the set of used diffs WITHOUT this row
                # For efficiency in the rebuild check, we can just check against diff_counts 
                # (which now excludes this row) and the new row's internal diffs.
                
                # Try to rebuild
                new_row = greedy_row_rebuild(set(diff_counts.keys()), best_local_max + 10, L, rng, attempts=3)
                
                if new_row is None:
                    # Revert
                    for d in old_diffs:
                        if d in diff_counts:
                            if diff_counts[d] == 1:
                                collision_count += 1
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    rows[r_idx] = old_row
                    row_diffs[r_idx] = old_diffs
                    continue
                
                new_diffs = get_diffs(new_row)
                
                # Add new diffs
                for d in new_diffs:
                    if d in diff_counts:
                        if diff_counts[d] == 1:
                            collision_count += 1
                        diff_counts[d] += 1
                    else:
                        diff_counts[d] = 1
                
                # Update row
                rows[r_idx] = new_row
                row_diffs[r_idx] = new_diffs
                
                new_sum_val = current_sum_val - sum(old_row) + sum(new_row)
                new_cost = collision_count * COLLISION_WEIGHT + new_sum_val
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_sum_val = new_sum_val
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_rows = [r[:] for r in rows]
                        best_local_max = max(r[-1] for r in best_local_rows)
                else:
                    delta = new_cost - current_cost
                    if delta == 0:
                        accept = rng.random() < 0.5
                    else:
                        if delta < 500:
                            p = math.exp(-delta / T)
                            accept = rng.random() < p
                        else:
                            accept = False
                    
                    if not accept:
                        # Revert
                        for d in new_diffs:
                            cnt = diff_counts[d]
                            if cnt == 2:
                                collision_count -= 1
                            diff_counts[d] = cnt - 1
                            if cnt - 1 == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            if d in diff_counts:
                                if diff_counts[d] == 1:
                                    collision_count += 1
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        rows[r_idx] = old_row
                        row_diffs[r_idx] = old_diffs
            
            T *= cooling
            if T < 1.0:
                T = 1.0

        return best_local_rows, best_local_max, best_local_cost

    # Main Loop
    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)
    
    # Run SA to improve
    # We give it most of the time
    sa_time = deadline - time.time() - 2.0
    if sa_time > 0:
        local_rows, local_max, local_cost = sa_optimize(best_rows, sa_time)
        # Check if valid (collisions == 0 implies cost < COLLISION_WEIGHT)
        if local_cost < COLLISION_WEIGHT:
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max

    return best_rows
