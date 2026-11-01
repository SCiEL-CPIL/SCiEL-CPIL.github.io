"""Rewritten program: randomized greedy + conflict-directed local search for difference triangle sets."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _all_diffs(rows):
    """Return dict mapping difference -> count, plus the set of colliding (row, i, j) pairs."""
    from collections import defaultdict
    cnt = defaultdict(int)
    # First pass: count
    for r, row in enumerate(rows):
        L = len(row)
        for i in range(L):
            ai = row[i]
            for j in range(i + 1, L):
                cnt[row[j] - ai] += 1
    # Second pass: find colliding pairs
    colliding = []
    for r, row in enumerate(rows):
        L = len(row)
        for i in range(L):
            ai = row[i]
            for j in range(i + 1, L):
                d = row[j] - ai
                if cnt[d] > 1:
                    colliding.append((r, i, j))
    return cnt, colliding


def _row_diffs(row):
    out = []
    L = len(row)
    for i in range(L):
        ai = row[i]
        for j in range(i + 1, L):
            out.append(row[j] - ai)
    return out


def _greedy_row(used, k, rng, max_x=200):
    """Build one row greedily with randomization to escape local minima."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        # Try a window of candidates around x
        best_x = None
        # First try sequential
        for trial in range(max_x):
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                best_x = x
                break
        if best_x is None:
            # Fallback: random search
            for _ in range(200):
                cand = random.randint(x + 1, x + 100)
                new = {cand - a for a in row}
                if not (new & used):
                    best_x = cand
                    break
        if best_x is None:
            return None
        row.append(best_x)
        used |= {best_x - a for a in row[:-1]}
        x = best_x
    return row


def _construct_greedy(n_rows, k, rng, max_x=200):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_x)
        if row is None:
            return None
        rows.append(row)
    return rows


def _is_valid(rows):
    from collections import defaultdict
    cnt = defaultdict(int)
    for row in rows:
        for d in _row_diffs(row):
            cnt[d] += 1
    return all(c == 1 for c in cnt.values())


def _local_search(rows, n_rows, k, rng, time_budget):
    """Conflict-directed local search: pick colliding (row,i,j), try to fix by adjusting row."""
    best_scope = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    start = time.time()

    # Precompute current colliding pairs
    cnt, colliding = _all_diffs(rows)
    if not colliding:
        return best_rows, best_scope

    # Build a per-row, per-position conflict count
    from collections import defaultdict
    pos_conflict = defaultdict(int)
    for (r, i, j) in colliding:
        pos_conflict[(r, i)] += 1
        pos_conflict[(r, j)] += 1

    while time.time() - start < time_budget:
        # Pick the position with the most conflicts
        if not pos_conflict:
            break
        max_conf = max(pos_conflict.values())
        # Collect all positions with max conflict
        candidates = [p for p, c in pos_conflict.items() if c == max_conf]
        (r, idx) = rng.choice(candidates)

        # Try to adjust row r by changing entry at idx
        row = rows[r]
        # Determine allowed range for new value at idx
        # Must be > row[idx-1] if idx > 0, and < row[idx+1] if idx < len-1
        lo = row[idx - 1] + 1 if idx > 0 else 1
        hi = row[idx + 1] - 1 if idx < len(row) - 1 else row[-1] + 50

        # Collect differences this row currently contributes
        old_diffs = set(_row_diffs(row))

        best_new_val = None
        best_new_row = None
        best_new_scope = best_scope + 1000

        # Try a few candidate values
        trial_vals = set()
        # Small perturbations
        for delta in range(-5, 6):
            v = row[idx] + delta
            if lo <= v <= hi:
                trial_vals.add(v)
        # Random candidates in range
        for _ in range(30):
            v = rng.randint(lo, hi)
            trial_vals.add(v)
        # Try to keep scope small: prefer values <= best_scope
        for v in sorted(trial_vals):
            if v > best_scope + 20:
                continue
            new_row = list(row)
            new_row[idx] = v
            # Check row is strictly increasing
            if any(new_row[t] >= new_row[t+1] for t in range(len(new_row)-1)):
                continue
            new_diffs = set(_row_diffs(new_row))
            # Check for conflicts with other rows
            # Temporarily remove old diffs, add new
            new_cnt = dict(cnt)
            # Remove old
            for d in old_diffs:
                new_cnt[d] -= 1
                if new_cnt[d] == 0:
                    del new_cnt[d]
            # Add new
            for d in new_diffs:
                new_cnt[d] = new_cnt.get(d, 0) + 1
            # Check validity
            if all(c == 1 for c in new_cnt.values()):
                new_scope = max(max(new_row), best_scope if r != 0 else 0)
                # Actually scope is max over all rows
                other_max = max(max(rows[rr]) for rr in range(n_rows) if rr != r)
                new_scope = max(max(new_row), other_max)
                if new_scope < best_new_scope:
                    best_new_scope = new_scope
                    best_new_val = v
                    best_new_row = new_row
                    best_cnt = new_cnt

        if best_new_row is not None:
            # Apply the change
            rows[r] = best_new_row
            cnt = best_cnt
            # Recompute conflicts
            _, colliding = _all_diffs(rows)
            pos_conflict = defaultdict(int)
            for (rr, ii, jj) in colliding:
                pos_conflict[(rr, ii)] += 1
                pos_conflict[(rr, jj)] += 1
            cur_scope = max(max(row) for row in rows)
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [list(row) for row in rows]
        else:
            # No improvement, try a different random position
            continue

    return best_rows, best_scope


def construct(n_rows, k):
    start = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    rng = random.Random(12345)

    # Phase 1: Try multiple greedy constructions with different random seeds
    best_rows = None
    best_scope = float('inf')

    # First, deterministic greedy (original)
    rows = _construct_greedy(n_rows, k, rng, max_x=300)
    if rows is not None and _is_valid(rows):
        scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_scope = scope

    # Try randomized greedy multiple times
    attempts = 0
    while time.time() - start < deadline - 30.0:
        rng2 = random.Random(attempts * 7919 + 1)
        rows = _construct_greedy(n_rows, k, rng2, max_x=300)
        if rows is not None and _is_valid(rows):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
        attempts += 1
        if attempts >= 20:
            break

    if best_rows is None:
        # Fallback: return a trivial valid set (may have large scope)
        # Use the original greedy without validation check
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Phase 2: Local search to improve scope
    time_left = deadline - (time.time() - start) - 5.0
    if time_left > 2.0:
        # Run local search on the best solution
        improved_rows, improved_scope = _local_search(
            [list(r) for r in best_rows], n_rows, k, rng, time_left
        )
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope

    return best_rows

# EVOLVE-BLOCK-END
