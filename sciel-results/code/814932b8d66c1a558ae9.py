"""Difference triangle set (7,5): full-backtracking DFS with pruning, symmetry breaking,
and a hard deadline guard. Minimizes the largest entry (scope)."""
import time
import random

_DEADLINE_SECONDS = 138.0

def _now():
    return time.monotonic()

def _scope(rows):
    return max(max(r) for r in rows) if rows else 0

def _greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _search(n_rows, k, scope_limit, deadline, rng, min_row_max=0):
    """Full backtracking DFS. Returns a valid set of n_rows rows with all entries <=
    scope_limit and distinct global positive differences, or None."""
    used = set()
    rows = []
    diffs_per_row = k * (k + 1) // 2
    total_needed = n_rows * diffs_per_row
    if scope_limit < total_needed:
        return None

    def build_row(row, min_final_max):
        if _now() > deadline:
            return False
        if len(row) == k + 1:
            if row[-1] < min_final_max:
                return False
            rows.append(list(row))
            return True
        last = row[-1]
        for x in range(last + 1, scope_limit + 1):
            if _now() > deadline:
                return False
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if not ok:
                continue
            added = set(x - a for a in row)
            used |= added
            row.append(x)
            if build_row(row, min_final_max):
                return True
            row.pop()
            used -= added
        return False

    def build_row_index(idx, cur_max):
        if _now() > deadline:
            return False
        if idx == n_rows:
            return True
        if cur_max > scope_limit:
            return False
        row = [0]
        if build_row(row, cur_max):
            new_max = row[-1]
            if build_row_index(idx + 1, new_max):
                return True
            rows.pop()
        return False

    if build_row_index(0, min_row_max):
        return rows
    return None

def construct(n_rows, k):
    start = _now()
    deadline = start + _DEADLINE_SECONDS

    best_rows = _greedy_construct(n_rows, k)
    best_scope = _scope(best_rows)

    rng = random.Random(999)

    diffs_per_row = k * (k + 1) // 2
    counting_bound = n_rows * diffs_per_row

    scopes_to_try = []
    if 112 >= counting_bound:
        scopes_to_try.append(112)
    for s in range(111, counting_bound - 1, -1):
        scopes_to_try.append(s)
    if best_scope not in scopes_to_try and best_scope >= counting_bound:
        scopes_to_try.append(best_scope)

    for scope in scopes_to_try:
        if _now() > deadline - 3:
            break
        if scope >= best_scope:
            continue
        res = _search(n_rows, k, scope, deadline, rng)
        if res is not None:
            sc = _scope(res)
            if sc < best_scope:
                best_scope = sc
                best_rows = res

    return best_rows
