import time
import random
import math
from collections import Counter

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
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

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

def _diff_sum(diff_counts):
    return sum(diff_counts.keys())

def _fast_greedy(k, n_rows, rng, spread=False):
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

def _csp_build_row(k, used_mask, max_val, rng, time_limit):
    """Build a row avoiding used_mask (bitmask of used differences)."""
    start = time.time()
    best_row = None
    best_score = float('inf')

    for attempt in range(20):
        if time.time() - start > time_limit:
            break

        row = [0]
        local_mask = 0
        ok = True

        for j in range(1, k + 1):
            found = False
            min_val = row[-1] + 1
            max_val_candidate = max_val - (k - j)

            if max_val_candidate < min_val:
                ok = False
                break

            if attempt < 3:
                candidates = range(min_val, max_val_candidate + 1)
            else:
                start_offset = rng.randint(0, max(0, max_val_candidate - min_val))
                candidates = list(range(min_val + start_offset, max_val_candidate + 1))
                if min_val + start_offset > min_val:
                    candidates += list(range(min_val, min_val + start_offset))
                rng.shuffle(candidates)

            for v in candidates:
                new_mask = 0
                valid = True
                for a in row:
                    d = v - a
                    bit = 1 << d
                    if (used_mask & bit) or (local_mask & bit):
                        valid = False
                        break
                    new_mask |= bit
                if valid:
                    row.append(v)
                    local_mask |= new_mask
                    found = True
                    break

            if not found:
                ok = False
                break

        if ok:
            score = row[-1]
            if score < best_score:
                best_score = score
                best_row = list(row)
                if best_score <= k * (k + 1) // 2:
                    break

    return best_row

def _mask_from_counts(diff_counts):
    m = 0
    for d in diff_counts:
        m |= (1 << d)
    return m

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope

    diff_counts = _build_diff_counts(current_rows)
    current_diff_sum = _diff_sum(diff_counts)
    best_diff_sum = current_diff_sum

    def _cost(diff_counts, scope):
        collisions = sum(max(0, c - 1) for c in diff_counts.values())
        return scope + 1000 * collisions + 0.001 * _diff_sum(diff_counts)

    current_cost = _cost(diff_counts, current_scope)
    best_cost = current_cost

    if polish:
        T = 0.3
        cooling = 0.9999
        min_T = 0.005
    else:
        T = 6.0
        cooling = 0.9995
        min_T = 0.05

    for it in range(iters):
        if time.time() > deadline:
            break

        r = rng.random()

        if r < 0.35:
            # Row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            used_mask = _mask_from_counts(diff_counts)

            new_row = _csp_build_row(k, used_mask, current_scope, rng, 0.02)
            if new_row is None:
                new_row = _csp_build_row(k, used_mask, current_scope + 8, rng, 0.02)

            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue

            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            new_scope = _scope(current_rows)
            new_cost = _cost(diff_counts, new_scope)

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row

        elif r < 0.55:
            # Scope compression on max element
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

            new_scope = _scope(current_rows)
            new_cost = _cost(diff_counts, new_scope)

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
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
            lo = current_rows[i][j - 1] + 1
            hi = current_rows[i][j + 1] - 1 if j < k else current_scope
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])

            new_scope = _scope(current_rows)
            new_cost = _cost(diff_counts, new_scope)

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])

        T = max(T * cooling, min_T)

    return best_scope, best_rows

def _push_down_polish(rows, k, deadline, rng):
    """Greedily reduce each row's entries to minimum valid values."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(current_rows)

    diff_counts = _build_diff_counts(current_rows)

    # Process rows in descending max-entry order
    order = sorted(range(n_rows), key=lambda i: -current_rows[i][-1])

    for _pass in range(3):
        for i in order:
            if time.time() > deadline:
                break
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            used_mask = _mask_from_counts(diff_counts)

            # Try to build minimal row
            new_row = _csp_build_row(k, used_mask, best_scope, rng, 0.05)
            if new_row is not None:
                _add_row_diffs(diff_counts, new_row)
                current_rows[i] = new_row
                new_scope = _scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row

    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')

    # Phase 1: diverse greedy initializations
    init_deadline = start_time + 8.0
    candidates = []

    for trial in range(40):
        if time.time() > init_deadline:
            break
        spread = rng.random() < 0.3
        rows = _fast_greedy(k, n_rows, rng, spread=spread)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]

    if not candidates:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                x = len(used) + j + 1
                while any((x - row[i]) in used for i in range(len(row))):
                    x += 1
                for i in range(len(row)):
                    used.add(x - row[i])
                row.append(x)
            rows.append(row)
        return rows

    # Phase 1.5: CSP tightening of best initial
    if best_rows and time.time() < deadline - 30:
        csp_rows = [list(r) for r in best_rows]
        diff_counts = _build_diff_counts(csp_rows)
        csp_scope = _scope(csp_rows)

        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for i in range(n_rows):
                if time.time() > deadline - 30:
                    break
                old_row = list(csp_rows[i])
                _remove_row_diffs(diff_counts, old_row)
                used_mask = _mask_from_counts(diff_counts)
                new_row = _csp_build_row(k, used_mask, csp_scope, rng, 0.06)
                if new_row:
                    _add_row_diffs(diff_counts, new_row)
                    csp_rows[i] = new_row
                    new_scope = _scope(csp_rows)
                    if new_scope < csp_scope:
                        csp_scope = new_scope
                        improved = True
                else:
                    _add_row_diffs(diff_counts, old_row)
                    csp_rows[i] = old_row

        if _valid(csp_rows, k):
            actual = _scope(csp_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = csp_rows

    candidates.sort(key=lambda x: x[0])

    # Phase 2: SA on top candidates
    remaining = deadline - time.time()
    top_n = min(3, len(candidates))
    time_per = remaining / max(1, top_n)

    for idx in range(top_n):
        if time.time() > deadline - 20:
            break
        s, rows = candidates[idx]
        sa_deadline = time.time() + time_per * 0.8
        sa_scope, sa_rows = _sa_optimize(rows, k, 6000, rng, sa_deadline)

        if sa_rows is not None and _valid(sa_rows, k):
            actual = _scope(sa_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = sa_rows
                candidates.append((actual, sa_rows))
                candidates.sort(key=lambda x: x[0])

    # Phase 3: push-down polish
    if best_rows is not None and time.time() < deadline - 5:
        polish_deadline = time.time() + min(15, deadline - time.time() - 5)
        polish_scope, polish_rows = _push_down_polish(best_rows, k, polish_deadline, rng)
        if polish_rows is not None and _valid(polish_rows, k):
            actual = _scope(polish_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = polish_rows

    # Final safety polish with tight SA
    if best_rows is not None and time.time() < deadline - 2:
        final_deadline = deadline - 1.0
        fin_scope, fin_rows = _sa_optimize(best_rows, k, 3000, rng, final_deadline, polish=True)
        if fin_rows is not None and _valid(fin_rows, k):
            actual = _scope(fin_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = fin_rows

    return best_rows
