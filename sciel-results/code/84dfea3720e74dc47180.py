import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 145.0
    MAX_DIFF = 200  # Safe upper bound for differences

    # Precompute bit masks for differences
    diff_bit = [1 << i for i in range(MAX_DIFF + 1)]

    def get_mask(diffs):
        mask = 0
        for d in diffs:
            if d <= MAX_DIFF:
                mask |= diff_bit[d]
        return mask

    def is_valid(rows):
        used = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(seed, M):
        rng = random.Random(seed)
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                found = False
                candidates = list(range(last + 1, M + 1))
                rng.shuffle(candidates)
                for x in candidates:
                    diffs = [x - a for a in row]
                    mask = get_mask(diffs)
                    if used_mask & mask:
                        continue
                    row.append(x)
                    used_mask |= mask
                    found = True
                    break
                if not found:
                    return None
            rows.append(row)
        return rows

    def backtrack_construct(seed, M, time_limit):
        rng = random.Random(seed)
        start_t = time.time()

        def solve(rows, used_mask, row_idx, row):
            if time.time() > start_t + time_limit:
                return None

            if row_idx == n_rows:
                return rows

            if len(row) == k + 1:
                return solve(rows + [row], used_mask, row_idx + 1, [0])

            last = row[-1]
            candidates = list(range(last + 1, M + 1))
            rng.shuffle(candidates)

            for x in candidates:
                diffs = [x - a for a in row]
                mask = get_mask(diffs)
                if used_mask & mask:
                    continue

                result = solve(rows, used_mask, row_idx, row + [x])
                if result is not None:
                    return result

            return None

        return solve([], 0, 0, [0])

    # Phase 1: Greedy to get a valid baseline
    best_rows = None
    best_scope = float('inf')

    for M in [120, 125, 130]:
        if time.time() > deadline - 10:
            break
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            rows = greedy_construct(seed, M)
            if rows:
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows

    if best_rows is None:
        # Fallback: try larger M
        for M in range(135, 160, 5):
            if time.time() > deadline - 5:
                break
            for seed in range(5):
                if time.time() > deadline - 2:
                    break
                rows = greedy_construct(seed, M)
                if rows:
                    scope = max(max(r) for r in rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = rows

    if best_rows is None:
        # Last resort: deterministic greedy
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                for x in range(last + 1, 200):
                    diffs = [x - a for a in row]
                    mask = get_mask(diffs)
                    if used_mask & mask:
                        continue
                    row.append(x)
                    used_mask |= mask
                    break
            rows.append(row)
        return rows

    # Phase 2: Try to improve with backtracking for smaller M
    # Start from min(112, best_scope - 1) and go down
    for M in range(min(112, best_scope - 1), 100, -1):
        if time.time() > deadline - 10:
            break
        time_left = deadline - time.time()
        time_alloc = min(8.0, time_left * 0.3)
        if time_alloc < 1.0:
            break

        for seed in range(10):
            if time.time() > deadline - 5:
                break
            t_limit = time_alloc / 10.0
            if t_limit < 0.5:
                break

            rows = backtrack_construct(seed, M, t_limit)
            if rows and is_valid(rows):
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                    if scope == M:
                        break  # Can't do better for this M

    return best_rows
