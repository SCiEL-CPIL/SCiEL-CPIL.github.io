import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def time_up():
        return time.time() > deadline

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def get_max(rows):
        return max(max(r) for r in rows)

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
    current_max = max(max(r) for r in rows)

    # Improved Rebuild Row: Collects candidates and uses weighted selection
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, polish_mode=False):
        row = [0]
        local_used = set()
        
        # In polish mode, we are stricter and faster
        max_candidates_to_check = 10 if polish_mode else 30
        checked = 0
        
        for step in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    # Score: primarily x, but we can add a small penalty for large gaps if needed
                    # Since max_new_diff is x, score is effectively x
                    valid_candidates.append((x, new_diffs))
                    checked += 1
                    if checked >= max_candidates_to_check:
                        break
            
            if not valid_candidates:
                return None
            
            # Sort by value (x) to prioritize smaller numbers
            valid_candidates.sort(key=lambda t: t[0])
            
            # Select from top few with weighted probability
            n_top = min(5, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]
            
            # Weights favor smaller x
            # Using 1/(x+1) creates a strong preference for small x
            weights = [1.0 / (c[0] + 1) for c in top_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = top_candidates[0]
            for c, w in zip(top_candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break
            
            x, new_diffs = chosen
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    # Two-row rebuild using the improved single-row rebuild
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit, polish_mode=False):
        row_a = rebuild_row(k, forbidden_diffs, rng, max_val_limit, polish_mode)
        if row_a is None:
            return None, None
        
        diffs_a = get_diffs(row_a)
        forbidden_b = forbidden_diffs | diffs_a
        
        row_b = rebuild_row(k, forbidden_b, rng, max_val_limit, polish_mode)
        if row_b is None:
            return None, None
            
        return row_a, row_b

    rng = random.Random(42)
    sa_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    no_improve_count = 0

    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Polish Mode: Last 15 seconds, only accept improvements
        polish_mode = remaining < 15.0
        
        if polish_mode:
            target_max = current_max - 1
            if target_max <= k:
                target_max = current_max # Can't go lower than k effectively for k+1 items starting at 0? 
                # Actually min max is k if 0..k. But diffs must be distinct.
                # For k=8, min max is likely > 8.
            accept_neutral = False
        else:
            target_max = current_max - 1
            if target_max <= k:
                target_max = current_max
            # Accept neutral moves with some probability to escape local minima
            accept_neutral = rng.random() < (0.2 * sa_temp)

        # Decide single vs multi-row rebuild
        # Use multi-row less often in polish mode to save time? Or more often to try harder?
        # Multi-row is more expensive. In polish, we want fast, reliable improvements.
        use_multi = (not polish_mode) and (n_rows >= 2) and (rng.random() < 0.3)
        
        if use_multi:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows
            
            old_row_i = rows[i]
            old_row_j = rows[j]
            old_diffs_i = get_diffs(old_row_i)
            old_diffs_j = get_diffs(old_row_j)
            
            used_diffs.difference_update(old_diffs_i)
            used_diffs.difference_update(old_diffs_j)
            
            new_row_i, new_row_j = rebuild_two_rows(k, used_diffs, rng, target_max, polish_mode)
            
            if new_row_i is None or new_row_j is None:
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
                continue
            
            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)
            
            if (new_diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (new_diffs_i & new_diffs_j):
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
                continue
            
            rows[i] = new_row_i
            rows[j] = new_row_j
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs_i)
                used_diffs.update(new_diffs_j)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                sa_temp = min(sa_temp * 1.1, 1.0)
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs_i)
                    used_diffs.update(new_diffs_j)
                    no_improve_count += 1
                else:
                    rows[i] = old_row_i
                    rows[j] = old_row_j
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    no_improve_count += 1
                    sa_temp = max(sa_temp * cooling_rate, min_temp)
        else:
            i = rng.randint(0, n_rows - 1)
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            used_diffs.difference_update(old_diffs)
            
            new_row = rebuild_row(k, used_diffs, rng, target_max, polish_mode)
            
            if new_row is None:
                used_diffs.update(old_diffs)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
                continue
            
            new_diffs = get_diffs(new_row)
            
            if new_diffs & used_diffs:
                used_diffs.update(old_diffs)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
                continue
            
            rows[i] = new_row
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                sa_temp = min(sa_temp * 1.1, 1.0)
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs)
                    no_improve_count += 1
                else:
                    rows[i] = old_row
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
                    sa_temp = max(sa_temp * cooling_rate, min_temp)

        # Lower bound check
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    return best_rows

# EVOLVE-BLOCK-END
