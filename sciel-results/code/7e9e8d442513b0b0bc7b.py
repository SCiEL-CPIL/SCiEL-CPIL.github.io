import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    best_result = None
    best_scope = float('inf')

    # Get initial greedy solution
    greedy = _greedy(n_rows, k)
    if greedy:
        best_result = greedy
        best_scope = max(max(r) for r in greedy)

    # Try to improve with backtracking
    for seed in range(100):
        if time.time() > deadline - 3.0:
            break

        random.seed(seed)
        result, scope = _search(n_rows, k, deadline - 3.0, best_scope)

        if result is not None and scope < best_scope:
            best_scope = scope
            best_result = result

    return best_result


def _greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
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


def _search(n_rows, k, deadline, upper_bound):
    used = set()
    rows = []
    result = [None]

    def dfs(row_idx, row, row_diffs):
        if time.time() > deadline:
            return False

        if row_idx == n_rows:
            scope = max(max(r) for r in rows)
            if scope < upper_bound:
                result[0] = [r[:] for r in rows]
            return True

        if len(row) == k + 1:
            rows.append(row[:])
            ok = dfs(row_idx + 1, [0], set())
            rows.pop()
            return ok

        last = row[-1]
        start = last + 1
        end = min(start + 100, upper_bound)

        candidates = list(range(start, end + 1))
        if len(candidates) > 5:
            random.shuffle(candidates[5:])

        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used or new_diffs & row_diffs:
                continue

            row.append(x)
            if dfs(row_idx, row, row_diffs | new_diffs):
                return True
            row.pop()

        return False

    dfs(0, [0], set())

    if result[0] is not None:
        return result[0], max(max(r) for r in result[0])
    return None, float('inf')
