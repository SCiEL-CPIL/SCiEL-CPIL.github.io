import random
import time
from collections import Counter

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    def get_max(rows):
        return max(max(r) for r in rows)

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, pressure_weight=1.0):
        """
        Builds a row using a hybrid strategy:
        - Collect valid candidates
        - Score each candidate by a combination of its value and the max new difference it introduces
        - Prefer candidates with low score (small value, low max new diff)
        """
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates up to max_val_limit
            # Limit search to avoid excessive time; we'll collect up to a reasonable number
            max_candidates_to_check = 50
            checked = 0
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    # Score: combination of x and max new diff
                    max_new_diff = max(new_diffs)
                    # Pressure weight favors keeping the max difference low, which helps fit in the limit
                    score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= max_candidates_to_check:
                        break
            
            if not valid_candidates:
                return None
            
            # Sort by score and pick from top few with some randomness
            valid_candidates.sort(key=lambda t: t[2])
            
            # Pick from top min(5, len) candidates with weighted probability favoring lower scores
            n_top = min(5, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]
            
            # Weighted random selection: lower score = higher probability
            # Using a simple inverse score weighting
            weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
            total_w = sum(weights)
            if total_w == 0:
                continue
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = top_candidates[0]
            for c, w in zip(top_candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break
            
            x, new_diffs, _ = chosen
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit, row_i, row_j):
        """
        Try to rebuild rows i and j simultaneously.
        Returns (new_row_i, new_row_j) or (None, None) if failed.
        """
        # First, build row i with forbidden_diffs
        row_i_new = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_i_new is None:
            return None, None
        
        diffs_i = get_diffs(row_i_new)
        
        # Now build row j with forbidden_diffs + diffs_i
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row(k, forbidden_j, rng, max_val_limit)
        if row_j_new is None:
            return None, None
        
        return row_i_new, row_j_new

    rng = random.Random(12345)
    
    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    temp = initial_temp
    
    no_improve_count = 0
    total_moves = 0
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Check if we should switch to polish mode (last 15 seconds)
        polish_mode = remaining < 15.0
        
        if polish_mode:
            # In polish mode, only accept improvements and use tighter limits
            target_max = current_max - 1
            accept_neutral = False
            max_candidates = 30
            pressure_weight = 1.0 # Maximize pressure to fit in tight limit
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.15 if temp > 0.1 else 0.05)
            max_candidates = 50
            pressure_weight = 0.5 # Balanced

        # Decide whether to do single or multi-row rebuild
        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)
        
        if use_multi:
            # Pick two distinct rows
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows
            
            old_row_i = rows[i]
            old_row_j = rows[j]
            old_diffs_i = get_diffs(old_row_i)
            old_diffs_j = get_diffs(old_row_j)
            
            # Remove both from used_diffs
            used_diffs.difference_update(old_diffs_i)
            used_diffs.difference_update(old_diffs_j)
            
            # Try to rebuild both
            new_row_i, new_row_j = rebuild_two_rows(
                k, used_diffs, rng, target_max if target_max > k else current_max, i, j
            )
            
            if new_row_i is None or new_row_j is None:
                # Failed, restore
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)
            
            # Verify no conflicts
            if (new_diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (new_diffs_i & new_diffs_j):
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            # Calculate new global max
            rows[i] = new_row_i
            rows[j] = new_row_j
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs_i)
                used_diffs.update(new_diffs_j)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)  # Reheat slightly
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs_i)
                    used_diffs.update(new_diffs_j)
                    no_improve_count += 1
                else:
                    # Revert
                    rows[i] = old_row_i
                    rows[j] = old_row_j
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    no_improve_count += 1
            
            temp *= cooling_rate
            total_moves += 1
        else:
            # Single row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            used_diffs.difference_update(old_diffs)
            
            # Try to rebuild with target_max (current_max - 1)
            new_row = rebuild_row(k, used_diffs, rng, target_max if target_max > k else current_max, pressure_weight)
            
            if new_row is None:
                # If failed to reduce, try to maintain current_max
                new_row = rebuild_row(k, used_diffs, rng, current_max, pressure_weight)
            
            if new_row is None:
                # Failed to rebuild at all
                used_diffs.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs = get_diffs(new_row)
            
            if new_diffs & used_diffs:
                used_diffs.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            rows[i] = new_row
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs)
                    no_improve_count += 1
                else:
                    rows[i] = old_row
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
            
            temp *= cooling_rate
            total_moves += 1
        
        # Early termination if we hit the theoretical lower bound
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    return best_rows

# EVOLVE-BLOCK-END
