import time
import random
import math
from collections import Counter

_BEST = None
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


def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows


def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts


def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]


def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1


def _row_rebuild(row, k, diff_counts, rng, max_scope):
    best_row = None
    best_score = float('inf')

    for attempt in range(12):
        row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.3:
                start = x + rng.randint(1, 3)
            for t in range(start, max_scope + 1):
                conflicts = 0
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                if conflicts == 0:
                    row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
                elif attempt < 4 and conflicts <= 1 and rng.random() < 0.2:
                    row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok and len(row) == k + 1:
            score = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if diff_counts.get(d, 0) > 0:
                        score += 1
            if score < best_score:
                best_score = score
                best_row = list(row)
            if best_score == 0:
                break

    return best_row


def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope

    diff_counts = _build_diff_counts(current_rows)

    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions

    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost

    if polish:
        T = 0.3
        cooling = 0.99995
        min_T = 0.01
    else:
        T = 6.0
        cooling = 0.9997
        min_T = 0.1

    for it in range(iters):
        if time.time() > deadline:
            break

        r = rng.random()

        if r < 0.18:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)

            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 8)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue

            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            new_cost = get_cost(current_rows, diff_counts)

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row

        elif r < 0.38:
            # Scope compression: try to reduce the max element
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break

            if max_row_idx < 0:
                continue

            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]

            prev_val = current_rows[i][j - 1]
            if old_val <= prev_val + 1:
                continue
            new_val = rng.randint(prev_val + 1, old_val - 1)

            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])

            new_cost = get_cost(current_rows, diff_counts)

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])

        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)

            old_val = current_rows[i][j]
            prev_val = current_rows[i][j - 1]
            next_val = current_rows[i][j + 1] if j < k else current_scope + 10

            lo = prev_val + 1
            hi = next_val - 1

            if lo > hi:
                continue

            if rng.random() < 0.7:
                span = max(1, hi - lo)
                delta = int(rng.gauss(0, max(1, span * 0.25)))
                new_val = old_val + delta
            else:
                new_val = rng.randint(lo, hi)

            if new_val == old_val or new_val < lo or new_val > hi:
                continue

            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])

            new_cost = get_cost(current_rows, diff_counts)

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])

        T = max(T * cooling, min_T)

    return best_scope, best_rows


def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0

    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

    rng = random.Random(42)

    candidates = []

    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))

    for _ in range(5):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)

    candidates.sort(key=_scope)

    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE - 10:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.35
        elif idx == 1:
            sa_deadline = time.time() + remaining * 0.30
        else:
            sa_deadline = time.time() + remaining * 0.25
        iters = max(int((sa_deadline - time.time()) * 8000), 2000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)

    if _BEST is not None and time.time() < _DEADLINE - 15:
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 10:
            polish_deadline = time.time() + remaining_time * 0.6
            polish_iters = max(int((polish_deadline - time.time()) * 10000), 500)
            s, rows = _sa_optimize(_BEST[1], k, polish_iters, rng, polish_deadline, polish=True)
            consider(rows)

            if time.time() < _DEADLINE - 5:
                final_deadline = _DEADLINE - 3
                final_iters = max(int((final_deadline - time.time()) * 12000), 200)
                s, rows = _sa_optimize(_BEST[1], k, final_iters, rng, final_deadline, polish=True)
                consider(rows)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
