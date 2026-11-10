import time
import random
import math
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 145  # Safety margin before 150s limit

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None, randomize=False, rng=None):
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row(row, used_set, cap, row_deadline, randomize=False, rng=None):
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None

        candidates = list(range(last + 1, upper + 1))
        if randomize and rng is not None:
            rng.shuffle(candidates)

        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize, rng)
            if res is not None:
                return res
        return None

    def validate_solution(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return False
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True

    # Phase 1: Greedy Baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            # Fallback: try with randomization
            row = greedy_row(set(used), k, randomize=True)
        if row is None:
            # Should not happen for reasonable inputs, but be safe
            return [[0] * (k+1) for _ in range(n_rows)]
        rows.append(row)
        used |= row_diffs(row)

    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max

    # Phase 2: Simulated Annealing
    T = 15.0
    cooling = 0.997
    rng = random.Random(42)

    def generate_candidate(i, randomize=True, cap=None):
        if cap is None:
            cap = current_max
        # Calculate used differences excluding row i
        used_without_i = set(used)
        used_without_i.difference_update(row_diffs(rows[i]))

        row_deadline = min(deadline - 5, time.time() + 0.8)
        cand = dfs_row([0], used_without_i, cap, row_deadline, randomize=randomize, rng=rng)
        return cand

    # Main SA Loop
    no_improve_count = 0
    while time.time() < deadline - 15:
        if T < 0.1:
            break

        # Pick a random row to perturb
        i = rng.randint(0, n_rows - 1)

        # Try to generate a candidate
        # Sometimes allow slightly larger cap to explore
        cap = current_max
        if rng.random() < 0.1:
            cap = current_max + rng.randint(1, 5)

        cand = generate_candidate(i, randomize=True, cap=cap)

        if cand is not None:
            new_last = cand[-1]
            max_others = 0
            for j in range(n_rows):
                if j != i:
                    if rows[j][-1] > max_others:
                        max_others = rows[j][-1]

            new_global_max = max(max_others, new_last)
            delta = new_global_max - current_max

            if delta < 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                accept = rng.random() < prob

            if accept:
                # Update state
                used.difference_update(row_diffs(rows[i]))
                rows[i] = cand
                used |= row_diffs(cand)

                current_max = new_global_max

                if current_max < best_max:
                    best_max = current_max
                    best_rows = rows[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # If stuck for too long, restart with a different seed or perturb more
                if no_improve_count > 500:
                    no_improve_count = 0
                    # Randomly reset a few rows to break symmetry
                    for _ in range(rng.randint(1, 3)):
                        j = rng.randint(0, n_rows - 1)
                        used.difference_update(row_diffs(rows[j]))
                        new_row = greedy_row(set(used), k, randomize=True, rng=rng)
                        if new_row is not None:
                            rows[j] = new_row
                            used |= row_diffs(new_row)
                            current_max = max(r[-1] for r in rows)

        T *= cooling

    # Phase 3: Final Polish
    final_rows = best_rows[:]
    final_used = set()
    for r in final_rows:
        final_used |= row_diffs(r)

    # Try to shrink each row further
    for i in range(n_rows):
        if time.time() > deadline - 3:
            break
        cur_last = final_rows[i][-1]
        if cur_last == 0:
            continue

        used_without = set(final_used)
        used_without.difference_update(row_diffs(final_rows[i]))

        # Try to find a smaller row
        found = None
        row_deadline = min(deadline - 3, time.time() + 2.0)

        # Try caps from cur_last - 1 down, but limit to a few
        for cap in range(cur_last - 1, max(0, cur_last - 20), -1):
            if time.time() > deadline - 3:
                break
            cand = dfs_row([0], used_without, cap, row_deadline, randomize=False, rng=rng)
            if cand is not None:
                found = cand
                break

        if found is not None:
            final_used.difference_update(row_diffs(final_rows[i]))
            final_rows[i] = found
            final_used |= row_diffs(found)

    # Final validation
    if not validate_solution(final_rows):
        # Fallback to best_rows if final polish broke something (shouldn't happen)
        if validate_solution(best_rows):
            return best_rows
        # Last resort: return greedy baseline
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return final_rows
