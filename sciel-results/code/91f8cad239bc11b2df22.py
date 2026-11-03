import time
import random
import math
from collections import Counter

_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    ds = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            ds.add(row[j] - row[i])
    return ds

def _fast_greedy(k, n_rows, spread=False):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if used.isdisjoint(new):
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _dfs_row(k, used, M, rng, max_nodes, order_seed=None):
    """Backtrack a single row of k+1 ints (starts 0, increasing, all <= M)
    whose internal diffs avoid `used`. Returns list or None."""
    row = [0]
    diffset = set()
    # order of positions to fill: last-first tends to spread and is a good heuristic
    if order_seed is None:
        order = list(range(1, k + 1))
    else:
        order = list(range(1, k + 1))
        rng.shuffle(order)

    nodes = 0
    limit = max_nodes

    def rec(pos):
        nonlocal nodes
        if nodes > limit:
            return False
        if not order:
            # row built; verify strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            return True
        p = order[0]
        rest = order[1:]
        # determine bounds for position p
        # lower bound: must be > row[p-1] if p-1 already filled, else 1
        lo = 1
        if p - 1 < len(row) and row[p - 1] > 0:
            lo = row[p - 1] + 1
        # upper bound: must be < row[p+1] if filled, else M
        hi = M
        if p + 1 < len(row) and row[p + 1] > 0:
            hi = row[p + 1] - 1
        if lo > hi:
            return False
        # candidate values
        vals = range(lo, hi + 1)
        # try values that keep room for remaining lower positions
        for v in vals:
            nodes += 1
            if nodes > limit:
                return False
            newd = {v - a for a in row if a < v}
            if not newd.isdisjoint(diffset):
                continue
            if not newd.isdisjoint(used):
                continue
            row.append(v)
            diffset |= newd
            row[p] = v
            if rec(rest):
                return True
            row[p] = 0
            row.pop()
            diffset -= newd
        return False

    # seed row with zeros at all positions then fill
    row = [0] * (k + 1)
    row[0] = 0
    if rec(order):
        return row
    return None

def _dfs_build(k, n_rows, M, rng, per_row_nodes):
    """Build n_rows rows with max entry <= M via DFS. Returns rows or None."""
    used = set()
    rows = []
    for _ in range(n_rows):
        r = _dfs_row(k, used, M, rng, per_row_nodes)
        if r is None:
            return None
        rows.append(r)
        used |= _row_diffs(r)
    return rows

def _local_repair(rows, k, rng, deadline, max_iters=400000):
    """Reduce scope via single-element moves that never increase collisions."""
    n_rows = len(rows)
    current = [list(r) for r in rows]
    best = [list(r) for r in current]
    best_scope = _scope(current)
    cur_scope = best_scope

    diff_counts = Counter()
    for r in current:
        for d in _row_diffs(r):
            diff_counts[d] += 1

    def collisions():
        return sum(max(0, c - 1) for c in diff_counts.values())

    def remove_row(r):
        for d in _row_diffs(r):
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

    def add_row(r):
        for d in _row_diffs(r):
            diff_counts[d] = diff_counts.get(d, 0) + 1

    cur_col = collisions()
    it = 0
    while it < max_iters and time.time() < deadline:
        it += 1
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current[i][j]
        prev_val = current[i][j - 1]
        next_val = current[i][j + 1] if j < k else 10 ** 9
        lo = prev_val + 1
        hi = next_val - 1
        if lo > hi:
            continue
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
        remove_row(current[i])
        current[i][j] = new_val
        add_row(current[i])
        new_col = collisions()
        new_scope = _scope(current)
        # accept if it improves scope (with no more collisions) or reduces collisions
        if (new_scope < cur_scope and new_col <= cur_col) or new_col < cur_col:
            cur_scope = new_scope
            cur_col = new_col
            if cur_scope < best_scope:
                best_scope = cur_scope
                best = [list(r) for r in current]
        else:
            remove_row(current[i])
            current[i][j] = old_val
            add_row(current[i])
    return best_scope, best

def _dfs_optimize(k, n_rows, rng, deadline):
    """Try to build a valid set with scope M via DFS, descend M, then repair."""
    # upper bound: scope of a decent greedy
    g = _fast_greedy(k, n_rows, spread=False)
    hi = _scope(g)
    lo = 1
    best = None
    best_scope = hi
    # repair the greedy first as a fallback
    s, rr = _local_repair(g, k, rng, min(deadline, time.time() + 20), max_iters=60000)
    if s < best_scope:
        best_scope = s
        best = rr

    # binary-search-ish DFS: find smallest M that DFS can build, then repair
    # Start from hi and try to go down.
    # We'll attempt a range of M values.
    # Estimate a lower bound from counting: need n_rows * C(k+1,2) distinct diffs <= M
    total_diffs = n_rows * (k * (k + 1) // 2)
    lb = int(math.ceil(math.sqrt(2 * total_diffs)))
    lb = max(lb, k)

    # We'll scan M from a few above lb upward, but bounded by time.
    # Strategy: try M = best_scope, then try to improve via DFS at lower M.
    candidates = set()
    for M in range(max(lb, 1), hi + 1):
        candidates.add(M)
    # Focus: try M values near best_scope and below.
    ordered = sorted(candidates)
    # We'll try from low to high but stop once we find a buildable M, then repair.
    # To save time, sample: try every value but with a per-row node budget that
    # grows as M gets closer to feasibility.
    for M in ordered:
        if time.time() > deadline - 5:
            break
        # per-row node budget
        per_row = 200000
        rows = _dfs_build(k, n_rows, M, rng, per_row)
        if rows is not None:
            # repair to reduce scope further
            s, rr = _local_repair(rows, k, rng, min(deadline, time.time() + 30),
                                   max_iters=120000)
            if s < best_scope:
                best_scope = s
                best = rr
            # since we found buildable at M, we can stop searching lower (none lower
            # are in ordered below current that are untried? we go low->high so once
            # buildable, higher M won't help reduce scope). Break.
            break
    return best_scope, best if best is not None else g

def construct(n_rows, k):
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 140.0
    rng = random.Random(12345)

    # Phase 1: quick greedy + repair to get a safe baseline
    g = _fast_greedy(k, n_rows, spread=False)
    s0, rr0 = _local_repair(g, k, rng, min(_DEADLINE, t0 + 15), max_iters=40000)
    best_scope = s0
    best_rows = rr0

    # Phase 2: DFS structural search
    if time.time() < _DEADLINE - 10:
        s1, rr1 = _dfs_optimize(k, n_rows, rng, _DEADLINE - 5)
        if s1 < best_scope:
            best_scope = s1
            best_rows = rr1

    # Phase 3: final repair on best
    if time.time() < _DEADLINE - 3:
        s2, rr2 = _local_repair(best_rows, k, rng, _DEADLINE - 1, max_iters=100000)
        if s2 < best_scope:
            best_scope = s2
            best_rows = rr2

    # Ensure validity of returned
    if _valid(best_rows, k):
        return best_rows
    # fallback
    g2 = _fast_greedy(k, n_rows, spread=False)
    return g2
