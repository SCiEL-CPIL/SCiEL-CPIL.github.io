"""Difference triangle set (n_rows, k) - bounded backtracking with forward checking."""
import time
from math import comb

def _viable_mask(row, used, m):
    # bit i set iff adding mark i to `row` introduces no difference already in `used`
    # and no duplicate within the row (all a < i are distinct, so no intra-row dup)
    v = 0
    for i in range(1, m + 1):
        ok = True
        for a in row:
            d = i - a
            if (used >> d) & 1:
                ok = False
                break
        if ok:
            v |= (1 << i)
    return v

def _dfs(row, used, viab, target, rem, deadline, budget, state):
    if time.time() > deadline or budget[0] <= 0:
        return False
    budget[0] -= 1
    if rem == 0:
        return True
    # choose smallest viable mark
    if viab:
        i = (viab & -viab).bit_length() - 1
        new_used = used | 0
        # compute new differences
        for a in row:
            new_used |= (1 << (i - a))
        new_row = row + [i]
        nv = _viable_mask(new_row, new_used, target)
        if nv:
            if _dfs(new_row, new_used, nv, target, rem - 1, deadline, budget, state):
                state['rows'].append(new_row)
                return True
    return False

def _solve_scope(n_rows, k, m, deadline, budget):
    # build rows one by one; each row must fit within [0, m]
    max_d = m  # largest possible difference
    used = 0
    rows = []
    budget[0] = max(budget[0], 1)
    # We need a global budget; use a shared counter via closure
    def build_row(idx, row, used, target):
        if time.time() > deadline or budget[0] <= 0:
            return False
        budget[0] -= 1
        if len(row) == k + 1:
            rows.append(row)
            if idx == n_rows - 1:
                return True
            return build_row(idx + 1, [0], used, target)
        viab = _viable_mask(row, used, target)
        # try smallest first (keeps marks low -> helps keep scope tight)
        v = viab
        while v:
            i = (v & -v).bit_length() - 1
            v &= v - 1
            new_used = used
            for a in row:
                new_used |= (1 << (i - a))
            if build_row(idx, row + [i], new_used, target):
                return True
        return False
    if build_row(0, [0], used, m):
        return rows
    return None

def _lower_bound(n_rows, k):
    # need n_rows * C(k+1,2) distinct positive differences, each in [1, m]
    # smallest m that can hold that many: m >= count (since differences 1..m give m values)
    total = n_rows * comb(k + 1, 2)
    return total  # crude: need at least `total` distinct diffs, so m >= total

def construct(n_rows, k):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s

    # greedy fallback (fast, always valid)
    used_g, rows_g = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_g:
                row.append(x)
                used_g |= new
        rows_g.append(row)
    greedy_scope = max(max(r) for r in rows_g)

    best = rows_g
    best_scope = greedy_scope

    # counting lower bound
    lb = _lower_bound(n_rows, k)
    # also scope must be >= k (row has k+1 increasing ints starting at 0, so last >= k)
    lb = max(lb, k)

    # try to improve: search scopes from lb upward, bounded by greedy_scope
    budget = [200000000]  # node budget
    for m in range(lb, greedy_scope + 1):
        if time.time() > deadline - 5.0:
            break
        budget[0] = 30000000
        res = _solve_scope(n_rows, k, m, deadline, budget)
        if res is not None:
            s = max(max(r) for r in res)
            if s < best_scope:
                best_scope = s
                best = res
            break  # found a valid set at this scope; it's the minimal found
        # if budget exhausted without success, try next (larger) scope
    return best
