"""Difference triangle set (n_rows, k) — greedy + simulated-annealing scope reduction."""
import time
import random
import math

# Public entry point (signature and return type unchanged).
def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin vs 150 s wall-clock kill
    seed = 1234567 + 101 * n_rows + 7 * k

    rows = _greedy(n_rows, k)
    if not rows:
        return rows
    best = [list(r) for r in rows]
    best_scope = _scope(rows)

    # If we already hit the counting bound, no need to search.
    if best_scope <= _counting_bound(n_rows, k):
        return best

    rng = random.Random(seed)

    # Phase 1: simple local search (downward shifts only).
    rows, scope = _local_search(rows, best_scope, rng, deadline, max_iter=100000)
    if scope < best_scope:
        best = [list(r) for r in rows]
        best_scope = scope
        if best_scope <= _counting_bound(n_rows, k):
            return best

    # Phase 2: simulated annealing with downward bias and occasional reheat.
    cur = [list(r) for r in best]
    cur_scope = best_scope
    T = 40.0
    T_min = 0.5
    cooling = 0.999
    down_bias = 0.70

    iters_since_improve = 0
    max_no_improve = 30000
    reheat_count = 0
    max_reheat = 4

    while time.time() < deadline:
        ri = rng.randrange(n_rows)
        ei = rng.randrange(k + 1)
        if ei == 0:
            continue
        val = cur[ri][ei]
        if rng.random() < down_bias:
            if val <= 1:
                continue
            lo = cur[ri][ei - 1] + 1
            hi = val - 1
            if hi < lo:
                continue
            new_val = rng.randrange(lo, hi + 1)
        else:
            lo = val + 1
            hi = max(val + rng.randrange(1, 30), val + 1)
            new_val = rng.randrange(lo, hi + 1)

        cur[ri][ei] = new_val
        new_scope = _scope(cur)
        delta = new_scope - cur_scope
        if delta <= 0 or rng.random() < _boltzmann(-delta / T):
            if new_scope < cur_scope:
                cur_scope = new_scope
                iters_since_improve = 0
                if cur_scope < best_scope:
                    best = [list(r) for r in cur]
                    best_scope = cur_scope
                    if best_scope <= _counting_bound(n_rows, k):
                        break
                T = max(T * 0.95, T_min)
            else:
                iters_since_improve += 1
        else:
            cur[ri][ei] = val
            iters_since_improve += 1

        T = max(T * cooling, T_min)

        if iters_since_improve >= max_no_improve:
            if reheat_count < max_reheat:
                T = 40.0
                reheat_count += 1
            iters_since_improve = 0

    return best


def _scope(rows):
    return max(max(r) for r in rows)


def _counting_bound(n_rows, k):
    return n_rows * k * (k + 1) // 2


def _greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _boltzmann(x):
    if x <= 0:
        return 1.0
    if x > 50:
        return 0.0
    return math.exp(-x)


def _local_search(rows, best_scope, rng, deadline, max_iter):
    cur = [list(r) for r in rows]
    cur_scope = best_scope
    n_rows = len(cur)
    k = len(cur[0]) - 1
    iters = 0
    while iters < max_iter and time.time() < deadline:
        iters += 1
        ri = rng.randrange(n_rows)
        ei = rng.randrange(1, k + 1)
        val = cur[ri][ei]
        if val <= 1:
            continue
        lo = cur[ri][ei - 1] + 1
        shift = rng.randrange(1, 4)
        new_val = val - shift
        if new_val < lo:
            continue
        cur[ri][ei] = new_val
        new_scope = _scope(cur)
        if new_scope < cur_scope:
            cur_scope = new_scope
        else:
            cur[ri][ei] = val
    return cur, cur_scope
