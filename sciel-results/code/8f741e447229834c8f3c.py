"""Difference Triangle Set (4,7) solver.

Goal: n_rows rows of k+1 increasing integers starting at 0, all positive differences
within rows distinct across rows, minimizing the largest entry.

Strategy (same family as parent – greedy + local improvement, but strengthened):
1. Fast deterministic greedy baseline.
2. Multiple randomized greedy restarts with different skip probabilities.
3. Aggressive local improvement: repeatedly try to decrease each element to its
   smallest feasible value using an efficient difference-set update.
4. Time-bounded loop with safety margin before the 150 s wall-clock limit.
"""
import time
import random
import sys


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng=None):
    """Deterministic / lightly randomized greedy ruler construction."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Optional random skip to diversify
            if rng.random() < 0.03:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _build_diff_set(rows):
    """Return the set of all positive differences present in rows."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _local_improve(rows, n_rows, k, rng, time_limit):
    """Iteratively try to reduce each element to the smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = _build_diff_set(work)
    current_max = _max_entry(work)
    L = k + 1

    def try_set(r_idx, pos, new_val):
        """Attempt to change work[r_idx][pos] to new_val. Returns True on success."""
        row = work[r_idx]
        # Strictly increasing check
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        # Collect old diffs involving pos
        old_diffs = set()
        for i in range(L):
            if i < pos:
                old_diffs.add(row[pos] - row[i])
            elif i > pos:
                old_diffs.add(row[i] - row[pos])
        # Collect new diffs involving pos
        new_diffs = set()
        for i in range(L):
            if i < pos:
                new_diffs.add(new_val - row[i])
            elif i > pos:
                new_diffs.add(row[i] - new_val)
        # All new diffs must be positive (guaranteed by increasing check) and not collide
        other = used_set - old_diffs
        if new_diffs & other:
            return False
        # Apply
        for d in old_diffs:
            used_set.discard(d)
        for d in new_diffs:
            used_set.add(d)
        row[pos] = new_val
        return True

    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Process positions from right to left within each row (greedy compaction)
        for r_idx in range(n_rows):
            for pos in range(L - 1, 0, -1):
                old_val = work[r_idx][pos]
                lower = work[r_idx][pos - 1] + 1
                upper = old_val - 1
                if upper < lower:
                    continue
                found = False
                # Try smallest feasible value first
                for new_val in range(lower, upper + 1):
                    if try_set(r_idx, pos, new_val):
                        if new_val < old_val:
                            improved = True
                            current_max = _max_entry(work)
                        found = True
                        break
                if not found:
                    # A few random attempts to escape local minima
                    for _ in range(10):
                        new_val = rng.randint(lower, upper)
                        if new_val == old_val:
                            continue
                        if try_set(r_idx, pos, new_val):
                            if new_val < old_val:
                                improved = True
                                current_max = _max_entry(work)
                            break
    return work


def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: stop searching at 130 s to leave room for final validation
    time_limit = start_time + 130

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Randomized greedy restarts
    rng = random.Random(12345)
    max_greedy_trials = 200
    for trial in range(max_greedy_trials):
        if time.time() > time_limit:
            break
        # Vary the skip probability
        skip_p = rng.random() * 0.15
        used = set()
        rows = []
        valid = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if rng.random() < skip_p:
                    x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
            if len(row) != k + 1:
                valid = False
                break
            rows.append(row)
        if valid and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local improvement (multiple passes)
    while time.time() < time_limit - 10:
        improved = _local_improve(best_rows, n_rows, k, rng, time_limit)
        if _check_valid(improved, k):
            imax = _max_entry(improved)
            if imax < best_max:
                best_rows = improved
                best_max = imax
            else:
                break
        else:
            break

    # 4) Final validation; fallback to pure greedy if something went wrong
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
