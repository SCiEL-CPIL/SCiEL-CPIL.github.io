import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110

    def elapsed():
        return time.time() - start_time

    def compute_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            for d in compute_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    def max_entry(rows):
        return max(max(row) for row in rows)

    def greedy_construct(seed, n_rows, k):
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def try_build_row(used_by_others, k, target_max):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > target_max:
                return None
            new_diffs = {x - a for a in row}
            if new_diffs & used_by_others:
                continue
            row.append(x)
        return row

    def local_search(rows, max_iter=500):
        rows = [row[:] for row in rows]
        current_max = max_entry(rows)

        for _ in range(max_iter):
            if elapsed() > deadline - 10:
                break

            improved = False
            for r in range(n_rows):
                used_by_others = set()
                for r2 in range(n_rows):
                    if r2 == r:
                        continue
                    used_by_others |= compute_diffs(rows[r2])

                old_max = rows[r][-1]
                for target in range(old_max - 1, 0, -1):
                    new_row = try_build_row(used_by_others, k, target)
                    if new_row is not None:
                        new_diffs = compute_diffs(new_row)
                        if not (new_diffs & used_by_others):
                            new_rows = [row[:] for row in rows]
                            new_rows[r] = new_row
                            new_max = max_entry(new_rows)
                            if new_max < current_max:
                                rows = new_rows
                                current_max = new_max
                                improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return rows, current_max

    # Initial solution
    best_rows = greedy_construct(0, n_rows, k)
    if not is_valid(best_rows):
        return best_rows
    best_max = max_entry(best_rows)

    # Try multiple greedy seeds
    for seed in range(1, 200):
        if elapsed() > deadline - 60:
            break
        rows = greedy_construct(seed, n_rows, k)
        if is_valid(rows):
            m = max_entry(rows)
            if m < best_max:
                best_max = m
                best_rows = [row[:] for row in rows]

    # Local search on best
    if elapsed() < deadline - 25:
        improved_rows, improved_max = local_search(best_rows, max_iter=800)
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows

    # More seeds with local search
    for seed in range(200, 500):
        if elapsed() > deadline - 5:
            break
        rows = greedy_construct(seed, n_rows, k)
        if is_valid(rows):
            m = max_entry(rows)
            if m < best_max:
                best_max = m
                best_rows = [row[:] for row in rows]
                if elapsed() < deadline - 15:
                    improved_rows, improved_max = local_search(best_rows, max_iter=300)
                    if improved_max < best_max:
                        best_max = improved_max
                        best_rows = improved_rows

    return best_rows
