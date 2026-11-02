"""Difference triangle set (4,7) - stochastic local search design."""
import time
import random
import math

_DEADLINE = 150.0
_MARGIN = 8.0  # seconds of safety before wall-clock kill


def _all_diffs(rows):
    """Return the set of all positive intra-row differences."""
    s = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
    return s


def _is_valid(rows, k):
    """Check each row has k+1 strictly increasing ints starting at 0, and all diffs unique."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        prev = -1
        for v in row:
            if v <= prev:
                return False
            prev = v
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy_row(used, k, rng):
    """Build one row greedily; optionally randomized by biasing small candidates."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _greedy_full(n_rows, k, rng):
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k, rng))
    return rows


def _local_search(rows, k, rng, deadline):
    """Simulated-annealing style local search to minimize scope.
    Moves:
      1. Replace a mark (not the 0) with a smaller positive value that keeps validity.
      2. Swap two marks between two rows (same position index) if valid.
    Objective: scope (max entry). We accept scope-decreasing moves always;
    occasionally accept increases to escape local minima.
    """
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    cur = [list(r) for r in rows]
    cur_scope = _scope(cur)
    cur_diffs = _all_diffs(cur)

    # Precompute row sizes
    m = k + 1

    def try_replace(row_idx, pos, new_val):
        """Attempt to replace cur[row_idx][pos] with new_val, returning (new_diffs, new_scope) or None."""
        if new_val <= 0:
            return None
        row = cur[row_idx]
        # strict increasing
        if pos > 0 and new_val <= row[pos - 1]:
            return None
        if pos < m - 1 and new_val >= row[pos + 1]:
            return None
        old_val = row[pos]
        if new_val == old_val:
            return None
        # recompute diffs for this row only, check global uniqueness
        # remove old diffs involving pos, add new
        old_row_diffs = set()
        for i in range(m):
            ai = row[i]
            for j in range(i + 1, m):
                d = row[j] - ai
                if i == pos or j == pos:
                    old_row_diffs.add(d)
        # build new row
        new_row = row[:]
        new_row[pos] = new_val
        new_row_diffs = set()
        for i in range(m):
            ai = new_row[i]
            for j in range(i + 1, m):
                new_row_diffs.add(new_row[j] - ai)
        # all new_row_diffs must not collide with (cur_diffs - old_row_diffs)
        other = cur_diffs - old_row_diffs
        if new_row_diffs & other:
            return None
        new_diffs = other | new_row_diffs
        # new scope
        ns = cur_scope
        if new_val > ns:
            ns = new_val
        # if old_val was the unique max, scope may drop
        if old_val == ns and new_val < ns:
            # recompute max
            ns = max(max(r) for r in cur)
            if new_val > ns:
                ns = new_val
        return (new_row, new_diffs, ns)

    def try_swap(row_i, row_j, pos):
        """Swap cur[row_i][pos] and cur[row_j][pos]; both rows must stay strictly increasing."""
        if row_i == row_j:
            return None
        ri = cur[row_i]
        rj = cur[row_j]
        vi = ri[pos]
        vj = rj[pos]
        if vi == vj:
            return None
        # check increasing for row_i after placing vj at pos
        if pos > 0 and vj <= ri[pos - 1]:
            return None
        if pos < m - 1 and vj >= ri[pos + 1]:
            return None
        if pos > 0 and vi <= rj[pos - 1]:
            return None
        if pos < m - 1 and vi >= rj[pos + 1]:
            return None
        # recompute diffs for both rows
        def row_diffs(r):
            s = set()
            for i in range(m):
                ai = r[i]
                for j in range(i + 1, m):
                    s.add(r[j] - ai)
            return s
        old_di = row_diffs(ri)
        old_dj = row_diffs(rj)
        other = cur_diffs - old_di - old_dj
        new_ri = ri[:]
        new_ri[pos] = vj
        new_rj = rj[:]
        new_rj[pos] = vi
        new_di = row_diffs(new_ri)
        new_dj = row_diffs(new_rj)
        if (new_di | new_dj) & other:
            return None
        new_diffs = other | new_di | new_dj
        ns = max(max(new_ri), max(new_rj), max(max(r) for r in cur if r is not ri and r is not rj))
        return (new_ri, new_rj, new_diffs, ns)

    # Annealing schedule
    start_time = time.time()
    total_steps = 0
    T = 12.0
    T_min = 0.3
    cool = 0.9995

    while time.time() < deadline:
        # pick a move
        r = rng.random()
        if r < 0.7:
            # replace move: try to lower a mark
            row_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, m)
            row = cur[row_idx]
            hi = row[pos] - 1
            lo = (row[pos - 1] + 1) if pos > 0 else 1
            if pos < m - 1:
                hi = min(hi, row[pos + 1] - 1)
            if hi < lo:
                continue
            # sample a few candidate values biased toward smaller
            cand = None
            for _ in range(3):
                cand = rng.randrange(lo, hi + 1)
                res = try_replace(row_idx, pos, cand)
                if res is not None:
                    break
            if res is None:
                continue
            new_row, new_diffs, ns = res
            d_scope = ns - cur_scope
            if d_scope <= 0 or rng.random() < math.exp(-d_scope / max(T, 1e-6)):
                # accept
                cur[row_idx] = new_row
                cur_diffs = new_diffs
                cur_scope = ns
                if ns < best_scope:
                    best_scope = ns
                    best = [list(x) for x in cur]
        else:
            # swap move
            row_i = rng.randrange(n_rows)
            row_j = rng.randrange(n_rows)
            if row_i == row_j:
                continue
            pos = rng.randrange(1, m)
            res = try_swap(row_i, row_j, pos)
            if res is None:
                continue
            new_ri, new_rj, new_diffs, ns = res
            d_scope = ns - cur_scope
            if d_scope <= 0 or rng.random() < math.exp(-d_scope / max(T, 1e-6)):
                cur[row_i] = new_ri
                cur[row_j] = new_rj
                cur_diffs = new_diffs
                cur_scope = ns
                if ns < best_scope:
                    best_scope = ns
                    best = [list(x) for x in cur]
        total_steps += 1
        T = max(T_min, T * cool)
    return best, best_scope


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    best_rows = None
    best_scope = None

    def consider(rows):
        nonlocal best_rows, best_scope
        if not _is_valid(rows, k):
            return
        s = _scope(rows)
        if best_scope is None or s < best_scope:
            best_scope = s
            best_rows = [list(r) for r in rows]

    # Phase 1: many randomized greedy restarts (fast, gives a solid baseline)
    while time.time() < deadline:
        rows = _greedy_full(n_rows, k, rng)
        consider(rows)
        # occasionally try row-order shuffling effect by rebuilding with different rng seeds
        if best_scope is not None and best_scope <= 120:
            # good enough baseline; break to local search
            break

    # Phase 2: local search from the best greedy solution
    if best_rows is not None:
        ls_rows, ls_scope = _local_search(best_rows, k, rng, deadline)
        if _is_valid(ls_rows, k):
            s = _scope(ls_rows)
            if best_scope is None or s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in ls_rows]

    # Phase 3: if time remains, a few more greedy restarts + short local search
    if time.time() < deadline:
        for _ in range(2000):
            if time.time() >= deadline:
                break
            rows = _greedy_full(n_rows, k, rng)
            consider(rows)
            if best_scope is not None and best_scope <= 115:
                break

        if best_rows is not None and time.time() < deadline:
            ls_rows, ls_scope = _local_search(best_rows, k, rng, deadline)
            if _is_valid(ls_rows, k):
                s = _scope(ls_rows)
                if best_scope is None or s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in ls_rows]

    # Fallback: deterministic greedy if somehow nothing valid
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            rows.append(_greedy_row(used, k, rng))
        best_rows = rows

    return best_rows
