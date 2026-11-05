import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Build a solid initial solution using a fast greedy algorithm.
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

    # Helper to compute max of rows
    def get_max(rows):
        return max(max(r) for r in rows)

    # Helper to compute diffs of a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # 2. Improved rebuild with difference-pressure candidate selection
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
            max_candidates_to_check = 50
            checked = 0
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    max_new_diff = max(new_diffs)
                    score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= max_candidates_to_check:
                        break
            
            if not valid_candidates:
                return None
            
            valid_candidates.sort(key=lambda t: t[2])
            
            n_top = min(5, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]
            
            weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
            total_w = sum(weights)
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

    # 3. Multi-row rebuild: try to rebuild 2 rows simultaneously
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit, row_i, row_j):
        row_i_new = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_i_new is None:
            return None, None
        
        diffs_i = get_diffs(row_i_new)
        
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row(k, forbidden_j, rng, max_val_limit)
        if row_j_new is None:
            return None, None
        
        return row_i_new, row_j_new

    # 4. Local search polish: try to reduce max by adjusting individual entries
    def local_polish(rows, used_diffs, max_limit, rng, max_iterations=1000):
        current_rows = [r[:] for r in rows]
        current_used = set(used_diffs)
        current_max_val = get_max(current_rows)
        
        improved = False
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
            max_row = current_rows[max_row_idx]
            max_entry = max_row[-1]
            
            if max_entry <= max_limit:
                break
            
            old_last = max_row[-1]
            old_diffs = get_diffs(max_row)
            
            current_used.difference_update(old_diffs)
            
            found = False
            for new_last in range(max_row[-2] + 1, max_entry):
                test_row = max_row[:-1] + [new_last]
                new_diffs = get_diffs(test_row)
                
                if not (new_diffs & current_used):
                    current_rows[max_row_idx] = test_row
                    current_used.update(new_diffs)
                    improved = True
                    found = True
                    break
            
            if not found:
                current_used.update(old_diffs)
                continue
            
            new_max = get_max(current_rows)
            if new_max <= max_limit:
                break
        
        if improved:
            return current_rows, current_used
        return None

    # 5. New: Simulated annealing on gap variables with collision penalty
    def anneal_gaps(rows, used_diffs, rng, max_iterations=5000):
        """
        Optimize using gap variables and a collision penalty.
        """
        current_rows = [r[:] for r in rows]
        current_used = set(used_diffs)
        current_max = get_max(current_rows)
        
        # Convert to gap representation
        gaps = []
        for row in current_rows:
            g = [row[i+1] - row[i] for i in range(k)]
            gaps.append(g)
        
        def gaps_to_rows(gaps):
            rows_out = []
            for g in gaps:
                row = [0]
                x = 0
                for gap in g:
                    x += gap
                    row.append(x)
                rows_out.append(row)
            return rows_out
        
        def compute_collision_penalty(gaps):
            """Compute penalty based on duplicate differences."""
            rows_out = gaps_to_rows(gaps)
            all_diffs = []
            for row in rows_out:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        all_diffs.append(row[j] - row[i])
            
            # Count duplicates
            from collections import Counter
            counts = Counter(all_diffs)
            penalty = sum(c * (c - 1) for c in counts.values())
            return penalty
        
        def compute_objective(gaps):
            """Objective: max entry + collision penalty."""
            rows_out = gaps_to_rows(gaps)
            max_entry = max(max(r) for r in rows_out)
            penalty = compute_collision_penalty(gaps)
            return max_entry + 10 * penalty
        
        current_obj = compute_objective(gaps)
        best_obj = current_obj
        best_gaps = [g[:] for g in gaps]
        
        temp = 5.0
        cooling = 0.99
        
        for it in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Pick a random gap to modify
            ri = rng.randint(0, n_rows - 1)
            gi = rng.randint(0, k - 1)
            
            old_gap = gaps[ri][gi]
            # Try to decrease or increase the gap
            if rng.random() < 0.7 and old_gap > 1:
                new_gap = max(1, old_gap - rng.randint(1, 3))
            else:
                new_gap = old_gap + rng.randint(1, 3)
            
            gaps[ri][gi] = new_gap
            new_obj = compute_objective(gaps)
            
            delta = new_obj - current_obj
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_gaps = [g[:] for g in gaps]
            else:
                gaps[ri][gi] = old_gap
            
            temp *= cooling
        
        # Convert best gaps back to rows and check validity
        best_rows = gaps_to_rows(best_gaps)
        
        # Verify validity
        all_diffs = []
        for row in best_rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    all_diffs.append(row[j] - row[i])
        
        if len(all_diffs) == len(set(all_diffs)):
            # Valid solution
            new_max = max(max(r) for r in best_rows)
            if new_max < current_max:
                return best_rows, new_max
        
        return None, current_max

    # Main Loop with simulated annealing
    rng = random.Random(12345)
    
    initial_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    temp = initial_temp
    
    no_improve_count = 0
    total_moves = 0
    
    # Try annealing gaps first
    if time.time() < deadline - 30:
        annealed_rows, annealed_max = anneal_gaps(rows, used_diffs, rng, max_iterations=3000)
        if annealed_rows is not None and annealed_max < current_max:
            rows = annealed_rows
            current_max = annealed_max
            best_rows = [r[:] for r in rows]
            used_diffs = set()
            for row in rows:
                used_diffs.update(get_diffs(row))
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        polish_mode = remaining < 15.0
        
        if polish_mode:
            target_max = current_max - 1
            accept_neutral = False
            max_candidates = 30
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.15 if temp > 0.1 else 0.05)
            max_candidates = 50

        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)
        
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
            
            new_row_i, new_row_j = rebuild_two_rows(
                k, used_diffs, rng, target_max if target_max > k else current_max, i, j
            )
            
            if new_row_i is None or new_row_j is None:
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)
            
            if (new_diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (new_diffs_i & new_diffs_j):
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
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
                temp = max(temp * 1.1, min_temp)
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
                
                temp *= cooling_rate
                total_moves += 1
        else:
            i = rng.randint(0, n_rows - 1)
            
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            used_diffs.difference_update(old_diffs)
            
            new_row = None
            target = target_max if target_max > k else current_max
            new_row = rebuild_row(k, used_diffs, rng, target)
            
            if new_row is None:
                new_row = rebuild_row(k, used_diffs, rng, current_max)
            
            if new_row is None:
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
        
        if no_improve_count > 200 and not polish_mode:
            num_to_rebuild = min(n_rows, rng.randint(2, 4))
            indices = rng.sample(range(n_rows), num_to_rebuild)
            
            old_rows = {}
            old_diffs_all = set()
            for idx in indices:
                old_rows[idx] = rows[idx]
                old_diffs_all.update(get_diffs(rows[idx]))
            
            used_diffs.difference_update(old_diffs_all)
            
            success = True
            new_rows = {}
            new_diffs_all = set()
            
            for idx in indices:
                target = target_max if target_max > k else current_max
                new_row = rebuild_row(k, used_diffs | new_diffs_all, rng, target)
                if new_row is None:
                    success = False
                    break
                new_rows[idx] = new_row
                new_diffs_all.update(get_diffs(new_row))
            
            if success and not (new_diffs_all & used_diffs):
                for idx in indices:
                    rows[idx] = new_rows[idx]
                used_diffs.update(new_diffs_all)
                new_global_max = get_max(rows)
                if new_global_max < current_max:
                    current_max = new_global_max
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0
                    temp = max(temp * 1.2, min_temp)
                else:
                    no_improve_count = 100
            else:
                for idx in indices:
                    rows[idx] = old_rows[idx]
                used_diffs.update(old_diffs_all)
                no_improve_count = 150
            
            temp *= cooling_rate
            total_moves += 1
        
        if remaining < 10.0:
            polished = local_polish(rows, used_diffs, current_max - 1, rng, max_iterations=500)
            if polished is not None:
                polished_rows, polished_used = polished
                polished_max = get_max(polished_rows)
                if polished_max < current_max:
                    rows = polished_rows
                    used_diffs = polished_used
                    current_max = polished_max
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0

        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    if time.time() < deadline - 2:
        polished = local_polish(rows, used_diffs, current_max - 1, rng, max_iterations=2000)
        if polished is not None:
            polished_rows, polished_used = polished
            polished_max = get_max(polished_rows)
            if polished_max < current_max:
                best_rows = polished_rows
                current_max = polished_max

    return best_rows

# EVOLVE-BLOCK-END
