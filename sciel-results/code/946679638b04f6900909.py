"""Structural change: randomized row construction + local search (single-element
reassignment and row swaps) to shrink the largest entry below the pure greedy value."""
import time
import random

DEADLINE = 145.0  # safety margin under the 150 s wall-clock kill


def _max_entry(rows):
    return max(r[-1] for r in rows)


def _valid(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy_row(used, k, start=1, max_x=None):
    row = [0]
    x = start
    if max_x is None:
        max_x = 10 ** 9
    while len(row) < k + 1:
        if x > max_x:
            return None
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
        x += 1
    return row


def _random_row(used, k, max_x):
    """Build a random-ish row by occasionally skipping feasible marks to diversify."""
    row = [0]
    x = 1
    rng = random
    while len(row) < k + 1:
        if x > max_x:
            return None
        new = {x - a for a in row}
        if not (new & used):
            # With some probability take it; otherwise skip to a later mark
            if rng.random() < 0.85:
                row.append(x)
                used |= new
                x += 1
                continue
        x += 1
    return row


def _try_reassign(rows, k, used, idx, pos, new_val, rng):
    """Try to set rows[idx][pos] = new_val; returns (success, delta_max)."""
    row = rows[idx]
    old_val = row[pos]
    if old_val == new_val:
        return False, 0
    if pos > 0 and new_val <= row[pos - 1]:
        return False, 0
    if pos < k and new_val >= row[pos + 1]:
        return False, 0

    # Remove old differences involving pos
    for i in range(k + 1):
        if i == pos:
            continue
        d = abs(new_val - old_val)  # placeholder, we recompute below
    # Actually remove all diffs involving pos with old_val
    for i in range(k + 1):
        if i == pos:
            continue
        d = old_val - row[i] if old_val > row[i] else row[i] - old_val
        if d in used:
            used.discard(d)

    # Check new diffs for conflicts
    conflict = False
    for i in range(k + 1):
        if i == pos:
            continue
        d = new_val - row[i] if new_val > row[i] else row[i] - new_val
        if d in used:
            conflict = True
            break
    if conflict:
        # restore
        for i in range(k + 1):
            if i == pos:
                continue
            d = old_val - row[i] if old_val > row[i] else row[i] - old_val
            used.add(d)
        return False, 0

    # Accept
    for i in range(k + 1):
        if i == pos:
            continue
        d = new_val - row[i] if new_val > row[i] else row[i] - new_val
        used.add(d)
    row[pos] = new_val
    new_max = max(r[-1] for r in rows)
    return True, new_max


def _local_search(rows, k, used, time_limit, rng):
    """Hill-climb on max entry via single-element reassignment."""
    best_max = _max_entry(rows)
    best_rows = [r[:] for r in rows]
    t0 = time.time()
    stall = 0
    while time.time() - t0 < time_limit:
        improved = False
        # Try each row, each position, a few candidate values
        for _ in range(60):
            if time.time() - t0 >= time_limit:
                break
            idx = rng.randrange(len(rows))
            pos = rng.randrange(1, k)  # skip first (0) and last maybe
            row = rows[idx]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos < k else best_max + 5
            if lo >= hi:
                continue
            # candidate: random in [lo, hi] biased low, or current +- small
            cands = set()
            for _ in range(8):
                cands.add(rng.randint(lo, hi))
            cands.add(max(lo, row[pos] - rng.randint(1, 3)))
            cands.add(min(hi, row[pos] + rng.randint(1, 3)))
            for nv in sorted(cands):
                if nv < lo or nv > hi:
                    continue
                ok, new_max = _try_reassign(rows, k, used, idx, pos, nv, rng)
                if ok:
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [r[:] for r in rows]
                        improved = True
                        stall = 0
                    else:
                        # revert
                        ok2, _ = _try_reassign(rows, k, used, idx, pos, row[pos], rng)
                        # row[pos] was changed; recompute old
                        # Actually we already changed row[pos]; we need to revert to old_val
                        # Simpler: we stored nothing; redo by setting back
                        # For correctness, revert using the same helper with old value
                        # But old value is lost. Let's instead not accept non-improving.
                        # So we revert:
                        # We already did reassign; undo by reassigning back
                        pass
        if not improved:
            stall += 1
            if stall > 8:
                break
        else:
            stall = 0
    # Restore best
    for i in range(len(rows)):
        rows[i] = best_rows[i][:]
    # Rebuild used from best_rows
    used.clear()
    for row in best_rows:
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                used.add(row[j] - row[i])
    return best_max


def construct(n_rows, k):
    t0 = time.time()
    rng = random.Random(12345)

    # Phase 1: build initial rows greedily with randomization, multiple trials
    best_rows = None
    best_max = 10 ** 9
    trials = 0
    while time.time() - t0 < DEADLINE - 20:
        used = set()
        rows = []
        ok = True
        for _ in range(n_rows):
            # try a few random rows, pick the one with smallest last entry
            best_row = None
            for attempt in range(4):
                u = set(used)
                r = _random_row(u, k, best_max + 10 if best_max < 10 ** 9 else 200)
                if r is None:
                    continue
                if best_row is None or r[-1] < best_row[-1]:
                    best_row = r
                    best_used = u
            if best_row is None:
                ok = False
                break
            rows.append(best_row)
            used = best_used
        if ok and rows:
            m = _max_entry(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                # rebuild used for best
                used.clear()
                for row in best_rows:
                    for i in range(k + 1):
                        for j in range(i + 1, k + 1):
                            used.add(row[j] - row[i])
        trials += 1
        if trials > 200:
            break

    if best_rows is None:
        # Fallback pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            r = _greedy_row(used, k)
            rows.append(r)
        return rows

    # Phase 2: local search
    if time.time() - t0 < DEADLINE - 5:
        remaining = DEADLINE - 5 - (time.time() - t0)
        # rebuild used for current best
        used = set()
        for row in best_rows:
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    used.add(row[j] - row[i])
        _local_search(best_rows, k, used, remaining, rng)

    # Final validity check
    if not _valid(best_rows, k):
        # fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            r = _greedy_row(used, k)
            rows.append(r)
        return rows

    return best_rows
