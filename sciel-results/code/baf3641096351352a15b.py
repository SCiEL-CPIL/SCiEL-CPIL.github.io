"""Difference triangle set (4,7) - stronger search.

Strategy: per-row greedy gives a baseline. Then improve via:
  1) Randomized greedy with many restarts (keep best by max entry).
  2) Local search / simulated-annealing style moves on the best row set
     (swap / move a single mark to a new value that keeps row valid
     and does not collide with any difference in other rows).
  3) A bounded backtracking "repair" on the largest row to try to
     shrink its maximum.

Deadline guard: track wall-clock and return the best found with margin.
"""

import random
import time

_DEADLINE = 150.0
_MARGIN = 10.0


def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s


def _valid_rows(rows):
    seen = set()
    for row in rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ai
                if d in seen:
                    return False
                seen.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_row(used, k, rng=None):
    """Build one row starting at 0, k+1 increasing marks, differences avoid `used`."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        ok = True
        for a in row:
            d = x - a
            if d in used:
                ok = False
                break
        if ok:
            row.append(x)
    return row


def _greedy_set(n_rows, k, rng=None):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        rows.append(row)
        for d in _row_diffs(row):
            used.add(d)
    return rows


def _randomized_greedy(n_rows, k, rng, max_x=400):
    """Greedy but at each step pick a random feasible next mark (bias to small)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # window of candidate marks to choose from
            lo = x + 1
            hi = min(max_x, lo + 25)
            best = None
            for cand in range(lo, hi + 1):
                ok = True
                for a in row:
                    if (cand - a) in used:
                        ok = False
                        break
                if ok:
                    if best is None:
                        best = cand
                    else:
                        # prefer smaller with some randomness
                        if rng.random() < 0.25:
                            best = cand
                        else:
                            # keep smaller with prob
                            if best > cand and rng.random() < 0.5:
                                best = cand
            if best is None:
                # expand window
                x = hi
                continue
            row.append(best)
            x = best
        rows.append(row)
        for d in _row_diffs(row):
            used.add(d)
    return rows


def _local_search(rows, n_rows, k, rng, time_limit):
    """Try single-mark moves/swaps that keep global validity and reduce max."""
    best = [list(r) for r in rows]
    best_score = _max_entry(best)
    start = time.time()

    def used_diffs(exclude_row=None):
        seen = set()
        for idx, row in enumerate(best):
            if idx == exclude_row:
                continue
            for i in range(len(row)):
                ai = row[i]
                for j in range(i + 1, len(row)):
                    seen.add(row[j] - ai)
        return seen

    # Precompute a reasonable search space for marks
    for it in range(200000):
        if time.time() - start > time_limit:
            break
        ridx = rng.randrange(n_rows)
        row = best[ridx]
        pos = rng.randrange(1, len(row))  # don't move the 0
        old = row[pos]
        # candidate new values
        cands = set()
        base = old
        for off in range(-15, 16):
            cands.add(base + off)
        for _ in range(20):
            cands.add(rng.randrange(1, base + 40))
        for nv in sorted(cands):
            if nv <= 0:
                continue
            # must keep row strictly increasing
            if pos > 0 and nv <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and nv >= row[pos + 1]:
                continue
            # build new row
            new_row = row[:]
            new_row[pos] = nv
            # check row internal diffs vs other rows' diffs
            other = used_diffs(exclude_row=ridx)
            ok = True
            for i in range(len(new_row)):
                ai = new_row[i]
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - ai
                    if d in other:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            new_score = _max_entry(best) if False else max(_max_entry(best), nv)
            # recompute max of this row
            row_max = max(new_row)
            others_max = 0
            for idx2, r2 in enumerate(best):
                if idx2 == ridx:
                    continue
                if max(r2) > others_max:
                    others_max = max(r2)
            new_score = max(row_max, others_max)
            if new_score < best_score:
                best[ridx] = new_row
                best_score = new_score
                break
    return best, best_score


def construct(n_rows, k):
    deadline = _DEADLINE - _MARGIN
    start = time.time()
    rng = random.Random(12345)

    # Baseline deterministic greedy
    best = _greedy_set(n_rows, k)
    best_score = _max_entry(best)

    # Randomized greedy restarts
    while time.time() - start < deadline - 20:
        cand = _randomized_greedy(n_rows, k, rng)
        sc = _max_entry(cand)
        if sc < best_score:
            best = cand
            best_score = sc

    # Local search polish
    if time.time() - start < deadline - 5:
        best, best_score = _local_search(best, n_rows, k, rng, deadline - time.time() - 5)

    # Final validity guard
    if not _valid_rows(best):
        best = _greedy_set(n_rows, k)

    return best
