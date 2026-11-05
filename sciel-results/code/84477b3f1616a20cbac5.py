import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    def time_up():
        return time.time() > deadline

    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # Fast greedy with backtracking limit for initial solution
    def greedy_construct(seed, max_backtrack=3):
        rng = random.Random(seed)
        used = set()
        rows = []

        def build_row(row, forbidden, local_used, max_val):
            if len(row) == k + 1:
                return True
            last = row[-1]
            candidates = []
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden):
                    candidates.append((x, new_diffs))
                if len(candidates) > 20:
                    break

            if not candidates:
                return False

            rng.shuffle(candidates)
            # Try candidates in random order
            for x, new_diffs in candidates:
                row.append(x)
                local_used_copy = local_used | new_diffs
                if build_row(row, forbidden, local_used_copy, max_val):
                    return True
                row.pop()

            return False

        # Increase max_val to ensure we find an initial solution
        # The optimal is around 134 for (4,7). 128 was too tight.
        max_val = k * (k + 1) // 2 + n_rows * 50 + 50
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            if not build_row(row, used, local_used, max_val):
                # If it fails with this max_val, try a larger one? 
                # Or just return None and let the outer loop handle it.
                return None
            rows.append(row)
            used.update(get_diffs(row))

        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None

    for seed in range(15):
        if time_up():
            break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used = used

    if best_rows is None:
        # Fallback: Construct a valid solution with a larger range
        # Use a greedy approach with a very large max_val and no randomization (or fixed)
        # to guarantee a solution exists and is found.
        # For (4,7), max 134 is optimal. Let's try up to 200.
        for seed in range(50):
            if time_up():
                break
            res = greedy_construct(seed + 1000)
            if res:
                rows, used = res
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used = used
                current_max = get_max(rows)
                break
        
        if best_rows is None:
            # Absolute last resort: Return a trivially valid structure if possible
            # This might have a large max, but it's valid.
            # We can use disjoint blocks if we can find 4 Sidon sets with disjoint diffs.
            # Given the constraints, the greedy above with max_val=200 should have succeeded.
            # If it didn't, we return an empty list to avoid crash, but the task requires 4 rows.
            # Let's force a valid construction by increasing max_val further in a final attempt.
            res = greedy_construct(9999, max_backtrack=10)
            if res:
                best_rows = [r[:] for r in res[0]]
            else:
                # If still None, we have a problem. 
                # However, for (4,7), a solution exists at 134. 
                # The greedy with max_val=200 should find it.
                # If this line is reached, it means the greedy is flawed.
                # Let's just return a list of 4 rows with 0s? No, must be increasing.
                # Return 4 rows of [0, 1, 3, 7, 12, 18, 25, 33] shifted?
                # Diffs of [0,1,3,7,12,18,25,33] are {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33}.
                # This is a Sidon set? 
                # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33.
                # Are they distinct? Yes.
                # But if we shift, the diffs remain the same. So we need 4 *different* Sidon sets with disjoint diffs.
                # This is hard to hardcode.
                # The greedy with max_val=200 is the best bet.
                return []

    # Lower bound check
    lb = max(k, n_rows * k * (k + 1) // 2)
    if best_max == lb:
        return best_rows

    rng = random.Random(42)
    current_max = best_max

    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, pressure_weight=1.0):
        row = [0]
        local_used = set()

        for step in range(k):
            last = row[-1]
            valid_candidates = []
            checked = 0

            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    max_new_diff = max(new_diffs)
                    score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= 30:
                        break

            if not valid_candidates:
                return None

            valid_candidates.sort(key=lambda t: t[2])
            n_top = min(8, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]

            weights = [1.0 / (1.0 + c[2] * 0.05) for c in top_candidates]
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

    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit):
        row_i_new = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_i_new is None:
            return None, None

        diffs_i = get_diffs(row_i_new)
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row(k, forbidden_j, rng, max_val_limit)
        if row_j_new is None:
            return None, None

        return row_i_new, row_j_new

    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.997
    min_temp = 0.005
    temp = initial_temp

    no_improve_count = 0
    total_moves = 0

    while not time_up():
        elapsed = time.time() - start_time
        remaining = deadline - elapsed

        polish_mode = remaining < 20.0

        if polish_mode:
            target_max = current_max - 1
            accept_neutral = False
            pressure_weight = 1.0
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.2 if temp > 0.1 else 0.08)
            pressure_weight = 0.7 + 0.3 * temp

        use_multi = (not polish_mode) and (rng.random() < 0.15) and (n_rows >= 2)

        if use_multi:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows

            old_row_i = current_rows[i]
            old_row_j = current_rows[j]
            old_diffs_i = get_diffs(old_row_i)
            old_diffs_j = get_diffs(old_row_j)

            current_used.difference_update(old_diffs_i)
            current_used.difference_update(old_diffs_j)

            new_row_i, new_row_j = rebuild_two_rows(
                k, current_used, rng, target_max if target_max > k else current_max
            )

            if new_row_i is None or new_row_j is None:
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue

            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)

            if (new_diffs_i & current_used) or (new_diffs_j & current_used) or (new_diffs_i & new_diffs_j):
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue

            current_rows[i] = new_row_i
            current_rows[j] = new_row_j
            new_global_max = get_max(current_rows)

            if new_global_max < current_max:
                current_used.update(new_diffs_i)
                current_used.update(new_diffs_j)
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.15, min_temp)
            else:
                if accept_neutral:
                    current_used.update(new_diffs_i)
                    current_used.update(new_diffs_j)
                    no_improve_count += 1
                else:
                    current_rows[i] = old_row_i
                    current_rows[j] = old_row_j
                    current_used.update(old_diffs_i)
                    current_used.update(old_diffs_j)
                    no_improve_count += 1

            temp *= cooling_rate
            total_moves += 1
        else:
            i = rng.randint(0, n_rows - 1)
            old_row = current_rows[i]
            old_diffs = get_diffs(old_row)

            current_used.difference_update(old_diffs)

            new_row = None
            if target_max > k:
                new_row = rebuild_row(k, current_used, rng, target_max, pressure_weight)

            if new_row is None:
                new_row = rebuild_row(k, current_used, rng, current_max, pressure_weight)

            if new_row is None:
                current_used.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue

            new_diffs = get_diffs(new_row)

            if new_diffs & current_used:
                current_used.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue

            current_rows[i] = new_row
            new_global_max = get_max(current_rows)

            if new_global_max < current_max:
                current_used.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.15, min_temp)
            else:
                if accept_neutral:
                    current_used.update(new_diffs)
                    no_improve_count += 1
                else:
                    current_rows[i] = old_row
                    current_used.update(old_diffs)
                    no_improve_count += 1

            temp *= cooling_rate
            total_moves += 1

    # Final polish: try to reduce max further with tighter constraints
    if not time_up():
        polish_deadline = min(deadline, time.time() + 12.0)
        while time.time() < polish_deadline:
            improved = False
            for i in range(n_rows):
                old_row = current_rows[i]
                old_diffs = get_diffs(old_row)
                current_used.difference_update(old_diffs)

                target_max = current_max - 1
                if target_max > k:
                    new_row = rebuild_row(k, current_used, rng, target_max, 1.0)
                    if new_row is not None:
                        new_diffs = get_diffs(new_row)
                        if not (new_diffs & current_used):
                            current_rows[i] = new_row
                            current_used.update(new_diffs)
                            current_max = get_max(current_rows)
                            best_rows = [r[:] for r in current_rows]
                            improved = True
                            break
                        else:
                            current_used.update(old_diffs)
                    else:
                        current_used.update(old_diffs)
                else:
                    current_used.update(old_diffs)

            if not improved:
                break

    return best_rows

# EVOLVE-BLOCK-END
