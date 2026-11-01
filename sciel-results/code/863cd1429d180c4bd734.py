"""Rewritten program: structural change to a randomized local-search / simulated-annealing
approach over the difference-set representation, with a hard wall-clock deadline guard."""
import time
import random
import sys
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    # Simple counting bound: largest entry m must satisfy m(m+1)/2 >= total_diffs
    m_min = 0
    while m_min * (m_min + 1) // 2 < total_diffs:
        m_min += 1
    lower_bound = m_min

    best_rows = None
    best_scope = None

    def get_scope(rows):
        return max(max(r) for r in rows)

    # Initial solution: greedy
    def greedy_init():
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

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    # Fast validation: returns (valid, scope) using early termination
    def fast_validate(rows):
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                if ri > scope:
                    scope = ri
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return (False, None)
                    seen.add(diff)
        return (True, scope)

    # Incremental move check: much faster than full validation
    # Precompute per-row diff lists for O(1) update
    def compute_row_diffs(row):
        """Return list of (i, j, diff) for all pairs in row."""
        res = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                res.append((i, j, row[j] - row[i]))
        return res

    # Local search with restarts
    max_restarts = 500
    steps_per_restart = 30000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start from best or random perturbation
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb: random row, random mark, random new value
            for _ in range(rng.randint(1, 4)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                old = cur[ri][pos]
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        valid, cur_scope = fast_validate(cur)
        if not valid:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Annealing
        T = max(0.05, 1.0 - restart / max_restarts)
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 25
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            valid, new_scope = fast_validate(cur)

            if valid:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                    cur_scope = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                cur[ri][pos] = old_val

            T *= 0.99995

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
