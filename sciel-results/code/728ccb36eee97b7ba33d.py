"""Difference triangle set: minimize the largest entry m(n_rows, k)."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _valid(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_one_row(used, k, start_hint=1, rng=None):
    """Build one row greedily. Returns (row, new_used, max_val) or None if stuck."""
    row = [0]
    x = start_hint
    while len(row) < k + 1:
        ok = False
        while True:
            new = set()
            conflict = False
            for a in row:
                d = x - a
                if d in used or d in new:
                    conflict = True
                    break
                new.add(d)
            if not conflict:
                row.append(x)
                used |= new
                ok = True
                break
            x += 1
            if x > 5000:  # safety
                return None
    return row, used, row[-1]

def _greedy_full(n_rows, k, order=None, start_hints=None, rng=None):
    """Build all rows. order: list of row indices to build in that order (permutation).
       start_hints: optional list of starting x for each row."""
    used = set()
    rows = [None] * n_rows
    if order is None:
        order = list(range(n_rows))
    for idx in order:
        hint = start_hints[idx] if start_hints else 1
        res = _greedy_one_row(used, k, start_hint=hint, rng=rng)
        if res is None:
            return None
        row, used, mx = res
        rows[idx] = row
    return rows

def construct(n_rows, k):
    deadline = time.time() + 110  # 10s safety margin under 120s
    rng = random.Random(12345)

    best_rows = None
    best_mx = float('inf')

    # --- Phase 1: plain greedy (deterministic baseline) ---
    res = _greedy_full(n_rows, k)
    if res is not None:
        mx = _max_entry(res)
        if mx < best_mx:
            best_mx = mx
            best_rows = res

    # --- Phase 2: randomized restarts with different row orders and start hints ---
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random permutation of row build order
        order = list(range(n_rows))
        rng.shuffle(order)
        # Random start hints (small perturbations)
        start_hints = [1] * n_rows
        for i in range(n_rows):
            start_hints[i] = 1 + rng.randint(0, 3)

        res = _greedy_full(n_rows, k, order=order, start_hints=start_hints, rng=rng)
        if res is not None:
            mx = _max_entry(res)
            if mx < best_mx:
                best_mx = mx
                best_rows = res

    if best_rows is None:
        # Fallback: should not happen, but return something valid-ish
        return [[0] + list(range(1, k + 1)) for _ in range(n_rows)]
    return best_rows
# EVOLVE-BLOCK-END
