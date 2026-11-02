"""Difference triangle set (7,5) - SA with row-level moves."""
import time
import random
import sys

def _all_diffs(row):
    n = len(row)
    return set(row[j] - row[i] for i in range(n) for j in range(i + 1, n))

def _validity(rows, k):
    seen = set()
    for r in rows:
        d = _all_diffs(r)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_row(used, k, start=1, deadline=None):
    row = [0]
    x = start
    while len(row) < k + 1:
        if deadline is not None and time.time() > deadline:
            return None
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
            x += 1
        else:
            x += 1
    return row

def _local_row_optimize(row, used_others, k, max_iters=2000, deadline=None):
    """Try to shrink this row's max by element-level moves, given the diffs used by OTHER rows."""
    best = list(row)
    best_max = best[-1]
    used_others = set(used_others)
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # pick a random index (1..k) to try to reduce
        idx = random.randint(1, k)
        cur = best[idx]
        # try values lower than cur
        for cand in range(max(best[idx-1] + 1, 1), cur):
            new_row = best[:idx] + [cand] + best[idx+1:]
            # check strictly increasing
            if any(new_row[i+1] <= new_row[i] for i in range(len(new_row)-1)):
                continue
            d = _all_diffs(new_row)
            if len(d) != k * (k + 1) // 2:
                continue
            if d & used_others:
                continue
            if new_row[-1] < best_max:
                best = new_row
                best_max = new_row[-1]
                break
    return best

def _random_row(used, k, max_val, deadline=None):
    """Build a random row within [0, max_val] avoiding used diffs."""
    row = [0]
    tries = 0
    while len(row) < k + 1:
        if deadline is not None and time.time() > deadline:
            return None
        tries += 1
        if tries > 5000:
            return None
        x = random.randint(row[-1] + 1, max_val)
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    rng = random.Random(12345)

    # --- Phase 1: build initial rows via greedy (fast, deterministic-ish) ---
    used = set()
    rows = []
    for _ in range(n_rows):
        if time.time() > deadline - 30:
            break
        r = _greedy_row(used, k, deadline=deadline - 30)
        if r is None:
            break
        rows.append(r)
        used |= _all_diffs(r)

    if len(rows) < n_rows:
        # fill remaining with random rows in a larger range
        for _ in range(len(rows), n_rows):
            if time.time() > deadline - 20:
                break
            r = _random_row(used, k, max_val=200, deadline=deadline - 20)
            if r is None:
                r = _greedy_row(used, k, deadline=deadline - 20)
            if r is None:
                break
            rows.append(r)
            used |= _all_diffs(r)

    if len(rows) < n_rows:
        return rows  # can't fill; return what we have (evaluator may reject, but avoid crash)

    # --- Phase 2: SA / iterative improvement ---
    best_rows = [list(r) for r in rows]
    best_max = max(r[-1] for r in best_rows)

    def current_used_except(ri):
        s = set()
        for j, r in enumerate(rows):
            if j == ri:
                continue
            s |= _all_diffs(r)
        return s

    # SA loop
    temp = 30.0
    min_temp = 0.5
    cool = 0.998
    step = 0
    while time.time() < deadline - 10:
        step += 1
        if temp <= min_temp:
            temp = min_temp
        # pick a random row to modify
        ri = rng.randint(0, n_rows - 1)
        used_others = current_used_except(ri)
        old_row = rows[ri]
        old_max = old_row[-1]

        # Try a few move types
        improved = False
        # Move type A: replace row with a fresh greedy row (starting from 1)
        if rng.random() < 0.3:
            new_row = _greedy_row(used_others, k, deadline=deadline - 10)
            if new_row is not None:
                new_max = new_row[-1]
                if new_max < old_max or rng.random() < (new_max - old_max) / temp:
                    rows[ri] = new_row
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                    improved = True
        # Move type B: local optimize the row
        if not improved and rng.random() < 0.4:
            new_row = _local_row_optimize(old_row, used_others, k, max_iters=300, deadline=deadline - 10)
            new_max = new_row[-1]
            if new_max < old_max:
                rows[ri] = new_row
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
                improved = True
        # Move type C: perturb one element
        if not improved:
            idx = rng.randint(1, k)
            cur = old_row[idx]
            # try a few candidate values
            candidates = []
            for delta in range(-15, 16):
                cand = cur + delta
                if cand <= old_row[idx-1] or (idx < k and cand >= old_row[idx+1]):
                    continue
                if cand < 1:
                    continue
                candidates.append(cand)
            rng.shuffle(candidates)
            for cand in candidates[:5]:
                new_row = old_row[:idx] + [cand] + old_row[idx+1:]
                d = _all_diffs(new_row)
                if len(d) != k * (k + 1) // 2:
                    continue
                if d & used_others:
                    continue
                new_max = new_row[-1]
                delta_obj = new_max - old_max
                if delta_obj < 0 or rng.random() < (1.0 / (1.0 + abs(delta_obj) / temp)):
                    rows[ri] = new_row
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                    improved = True
                    break
        temp *= cool

    # --- Phase 3: final tightening: repeatedly local-optimize each row ---
    for _ in range(5):
        if time.time() > deadline - 2:
            break
        for ri in range(n_rows):
            if time.time() > deadline - 2:
                break
            used_others = current_used_except(ri)
            new_row = _local_row_optimize(rows[ri], used_others, k, max_iters=500, deadline=deadline - 2)
            if new_row[-1] < rows[ri][-1]:
                rows[ri] = new_row
                m = max(r[-1] for r in rows)
                if m < best_max:
                    best_max = m
                    best_rows = [list(r) for r in rows]

    # Return the best found
    return best_rows
