import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            for i in range(1, k + 1):
                if r[i] <= r[i - 1]:
                    return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def compute_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row(used_diffs, max_val, rng, skip_prob=0.0):
        row = [0]
        local_diffs = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            found = False
            while x <= max_val:
                if rng.random() < skip_prob and x < max_val:
                    x += 1
                    continue
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        bad = True
                        break
                    new_diffs.append(d)
                if not bad:
                    row.append(x)
                    local_diffs.update(new_diffs)
                    found = True
                    break
                x += 1
            if not found:
                return None
        return row

    def interleave_greedy(M_limit, seed_val=0, skip_prob=0.0):
        """
        Interleaved greedy: extend the row with the smallest current maximum at each step.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        local_diffs = [set() for _ in range(n_rows)]

        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break

            target_row_idx = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[target_row_idx]
            ldiffs = local_diffs[target_row_idx]

            x = row[-1] + 1
            found = False
            while x <= M_limit:
                if rng.random() < skip_prob and x < max_val_check(x, M_limit):
                    x += 1
                    continue
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in ldiffs:
                        bad = True
                        break
                    new_diffs.append(d)
                if not bad:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    ldiffs.update(new_diffs)
                    found = True
                    break
                x += 1
            if not found:
                return None

        for r in rows:
            if len(r) != k + 1:
                return None
        return rows

    def max_val_check(x, M):
        return M

    def local_search_polish(rows, M_limit, time_budget):
        """
        Systematically minimize each element in each row to reduce scope.
        For each row, for each position from last to first, try to decrease the value.
        """
        if rows is None:
            return None, None

        rng = random.Random(42)
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        current = [r[:] for r in best]
        current_scope = best_scope
        deadline = time.time() + time_budget

        # Precompute used diffs
        used_diffs = get_used_diffs(current)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to reduce each row's elements from right to left
            for ri in range(n_rows):
                if time.time() > deadline:
                    break
                row = current[ri]
                old_diffs = compute_row_diffs(row)
                used_without = used_diffs - old_diffs

                # Try to minimize each element from last to first
                for idx in range(k, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[idx]
                    # Try to decrease this element
                    lower = row[idx - 1] + 1
                    upper = row[idx + 1] - 1 if idx < k else old_val - 1

                    if lower >= upper:
                        continue

                    # Try values from lower to old_val-1 (smallest first)
                    for new_val in range(lower, old_val):
                        if time.time() > deadline:
                            break
                        new_row = row[:]
                        new_row[idx] = new_val
                        new_diffs = compute_row_diffs(new_row)

                        # Check internal consistency (all diffs distinct)
                        if len(new_diffs) != diffs_per_row:
                            continue

                        # Check against other rows
                        if new_diffs & used_without:
                            continue

                        # Valid! Update
                        used_diffs = used_without | new_diffs
                        current[ri] = new_row
                        row = new_row
                        old_diffs = new_diffs
                        used_without = used_diffs - old_diffs
                        improved = True

                        # Recompute scope
                        other_max = 0
                        for i, r in enumerate(current):
                            if i != ri:
                                other_max = max(other_max, max(r))
                        new_scope = max(other_max, max(new_row))
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best = [r[:] for r in current]
                        current_scope = new_scope
                        break  # Move to next element

        return best, best_scope

    def simulated_annealing(rows, time_budget, seed_val=0):
        if rows is None:
            return None, None

        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        current = [r[:] for r in best]
        current_scope = best_scope

        used = get_used_diffs(current)
        deadline = time.time() + time_budget
        T = 20.0
        cooling = 0.998
        no_improve = 0
        max_no_improve = 500

        while time.time() < deadline and T > 0.05:
            ri = rng.randint(0, n_rows - 1)
            old_row = current[ri]
            old_diffs = compute_row_diffs(old_row)
            used_without = used - old_diffs

            max_mark = current_scope
            new_row = None

            for attempt in range(40):
                sp = rng.uniform(0.0, 0.4)
                new_row = greedy_row(used_without, max_mark, rng, skip_prob=sp)
                if new_row is not None:
                    break

            if new_row is None:
                T *= cooling
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T, 15.0)
                    no_improve = 0
                continue

            new_diffs = compute_row_diffs(new_row)
            if len(new_diffs) != diffs_per_row:
                T *= cooling
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T, 15.0)
                    no_improve = 0
                continue

            if new_diffs & used_without:
                T *= cooling
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T, 15.0)
                    no_improve = 0
                continue

            other_max = 0
            for i, r in enumerate(current):
                if i != ri:
                    other_max = max(other_max, max(r))
            new_scope = max(other_max, max(new_row))

            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                T = max(T, 15.0)

            T *= cooling

        return best, best_scope

    # --- Main Logic ---

    best_rows = None
    best_M = float('inf')

    # Phase 1: Interleaved greedy with increasing M
    start_M = lower_bound
    end_M = lower_bound + 40

    for M in range(start_M, end_M):
        if time.time() > deadline - 40:
            break
        for seed in range(5):
            if time.time() > deadline - 35:
                break
            rows = interleave_greedy(M, seed, skip_prob=0.0)
            if rows is not None and is_valid(rows):
                current_max = get_scope(rows)
                if current_max < best_M:
                    best_M = current_max
                    best_rows = [r[:] for r in rows]
                break
        if best_rows is not None and best_M <= M:
            break

    # Fallback if interleaved greedy failed
    if best_rows is None:
        M_huge = 300
        rows = interleave_greedy(M_huge, 0)
        if rows and is_valid(rows):
            best_rows = rows
            best_M = get_scope(rows)
        else:
            # Trivial solution
            best_rows = []
            base = 0
            for i in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    base += 100
                    row.append(base)
                best_rows.append(row)
            best_M = base

    # Phase 2: Local search polish (systematic minimization)
    time_left = deadline - time.time()
    if time_left > 5:
        polish_budget = min(time_left - 5, 40)
        if polish_budget > 0:
            polished_rows, polished_M = local_search_polish(best_rows, best_M, polish_budget)
            if polished_rows is not None and polished_M < best_M:
                best_M = polished_M
                best_rows = polished_rows

    # Phase 3: Simulated annealing
    time_left = deadline - time.time()
    if time_left > 5:
        sa_budget = min(time_left - 5, 50)
        if sa_budget > 0:
            improved_rows, improved_M = simulated_annealing(best_rows, sa_budget, seed_val=42)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows

            # Polish again after SA
            time_left = deadline - time.time()
            if time_left > 5:
                polish_budget2 = min(time_left - 5, 20)
                if polish_budget2 > 0:
                    polished_rows2, polished_M2 = local_search_polish(best_rows, best_M, polish_budget2)
                    if polished_rows2 is not None and polished_M2 < best_M:
                        best_M = polished_M2
                        best_rows = polished_rows2

            # Another SA run if time permits
            time_left = deadline - time.time()
            if time_left > 5:
                sa_budget2 = min(time_left - 5, 20)
                if sa_budget2 > 0:
                    improved_rows2, improved_M2 = simulated_annealing(best_rows, sa_budget2, seed_val=123)
                    if improved_rows2 is not None and improved_M2 < best_M:
                        best_M = improved_M2
                        best_rows = improved_rows2

                    # Final polish
                    time_left = deadline - time.time()
                    if time_left > 3:
                        polish_budget3 = min(time_left - 3, 10)
                        if polish_budget3 > 0:
                            polished_rows3, polished_M3 = local_search_polish(best_rows, best_M, polish_budget3)
                            if polished_rows3 is not None and polished_M3 < best_M:
                                best_M = polished_M3
                                best_rows = polished_rows3

    return best_rows


if __name__ == "__main__":
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", get_scope(rows))
        print("Valid:", is_valid(rows))
    else:
        print("No solution found")
