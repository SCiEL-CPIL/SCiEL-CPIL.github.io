import random
import time
import math
from typing import List, Set, Optional

# EVOLVE-BLOCK-START

def construct(n_rows: int, k: int) -> List[List[int]]:
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    def time_up() -> bool:
        return time.time() > deadline

    def get_diffs(row: List[int]) -> Set[int]:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows: List[List[int]]) -> int:
        return max(max(r) for r in rows)

    # Greedy construction with backtracking
    def greedy_construct(seed: int) -> Optional[tuple]:
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
                if len(candidates) > 30:
                    break

            if not candidates:
                return False

            rng.shuffle(candidates)
            for x, new_diffs in candidates:
                row.append(x)
                local_used_copy = local_used | new_diffs
                if build_row(row, forbidden, local_used_copy, max_val):
                    return True
                row.pop()
            return False

        # Estimate a reasonable max_val
        # For (4,7), optimal is 134, counting bound is 112
        max_val = max(150, k * (k + 1) // 2 + n_rows * 40 + 50)

        for _ in range(n_rows):
            row = [0]
            local_used = set()
            if not build_row(row, used, local_used, max_val):
                return None
            rows.append(row)
            used.update(get_diffs(row))

        return rows, used

    # Find initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None

    for seed in range(20):
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
        # Fallback with larger max_val
        for seed in range(50):
            if time_up():
                break
            res = greedy_construct(seed + 1000)
            if res:
                best_rows = [r[:] for r in res[0]]
                current_rows = res[0]
                current_used = res[1]
                break

        if best_rows is None:
            # Last resort
            res = greedy_construct(9999)
            if res:
                best_rows = [r[:] for r in res[0]]
            else:
                return []

    current_max = best_max
    rng = random.Random(42)

    # Rebuild a single row with pressure toward compactness
    def rebuild_row(row_k, forbidden_diffs, rng, max_val_limit, pressure=1.0):
        row = [0]
        local_used = set()

        for step in range(row_k):
            last = row[-1]
            valid_candidates = []

            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    max_new_diff = max(new_diffs)
                    score = x * (1.0 - pressure) + max_new_diff * pressure
                    valid_candidates.append((x, new_diffs, score))

            if not valid_candidates:
                return None

            valid_candidates.sort(key=lambda t: t[2])
            n_top = min(10, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]

            weights = [1.0 / (1.0 + c[2] * 0.03) for c in top_candidates]
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

        return row

    # Simulated annealing
    initial_temp = 1.5
    cooling_rate = 0.998
    min_temp = 0.003
    temp = initial_temp
    no_improve_count = 0

    while not time_up():
        elapsed = time.time() - start_time
        remaining = deadline - elapsed

        polish_mode = remaining < 15.0

        if polish_mode:
            target_max = current_max - 1
            accept_neutral = False
            pressure = 1.0
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.25 if temp > 0.1 else 0.1)
            pressure = 0.6 + 0.4 * temp

        # Occasionally rebuild two rows together
        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)

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

            new_row_i = rebuild_row(k, current_used, rng, target_max if target_max > k else current_max, pressure)
            if new_row_i is None:
                new_row_i = rebuild_row(k, current_used, rng, current_max, pressure)

            if new_row_i is None:
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                continue

            diffs_i_new = get_diffs(new_row_i)
            if diffs_i_new & current_used:
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                continue

            forbidden_j = current_used | diffs_i_new
            new_row_j = rebuild_row(k, forbidden_j, rng, target_max if target_max > k else current_max, pressure)
            if new_row_j is None:
                new_row_j = rebuild_row(k, forbidden_j, rng, current_max, pressure)

            if new_row_j is None:
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                continue

            diffs_j_new = get_diffs(new_row_j)
            if diffs_j_new & current_used or diffs_j_new & diffs_i_new:
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                continue

            current_rows[i] = new_row_i
            current_rows[j] = new_row_j
            new_global_max = get_max(current_rows)

            if new_global_max < current_max:
                current_used.update(diffs_i_new)
                current_used.update(diffs_j_new)
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.2, min_temp)
            else:
                if accept_neutral:
                    current_used.update(diffs_i_new)
                    current_used.update(diffs_j_new)
                    no_improve_count += 1
                else:
                    current_rows[i] = old_row_i
                    current_rows[j] = old_row_j
                    current_used.update(old_diffs_i)
                    current_used.update(old_diffs_j)
                    no_improve_count += 1

            temp *= cooling_rate

        else:
            i = rng.randint(0, n_rows - 1)
            old_row = current_rows[i]
            old_diffs = get_diffs(old_row)

            current_used.difference_update(old_diffs)

            new_row = rebuild_row(k, current_used, rng, target_max if target_max > k else current_max, pressure)
            if new_row is None:
                new_row = rebuild_row(k, current_used, rng, current_max, pressure)

            if new_row is None:
                current_used.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                continue

            new_diffs = get_diffs(new_row)
            if new_diffs & current_used:
                current_used.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                continue

            current_rows[i] = new_row
            new_global_max = get_max(current_rows)

            if new_global_max < current_max:
                current_used.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.2, min_temp)
            else:
                if accept_neutral:
                    current_used.update(new_diffs)
                    no_improve_count += 1
                else:
                    current_rows[i] = old_row
                    current_used.update(old_diffs)
                    no_improve_count += 1

            temp *= cooling_rate

    # Final polish: aggressive single-row reduction
    if not time_up():
        polish_deadline = min(deadline, time.time() + 8.0)
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
