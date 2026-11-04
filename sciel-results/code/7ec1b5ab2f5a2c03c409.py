"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # Counting lower bound on the largest entry (informational)
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    best_rows = None
    best_max = float('inf')

    # ---- Constructor: pure greedy (baseline) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # ---- Constructor: lookahead greedy (scores candidates by future flexibility) ----
    def lookahead_greedy(n_rows, k, rng):
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Gather a window of valid candidates
                window = max(30, 70 - 8 * row_idx)
                valid_candidates = []
                for candidate in range(x + 1, x + window + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= 12:
                            break
                if not valid_candidates:
                    candidate = x + 1
                    while True:
                        new_diffs = {candidate - a for a in row}
                        if not new_diffs & used:
                            valid_candidates.append(candidate)
                            break
                        candidate += 1

                # Score each candidate by how many options it leaves for the next step
                best_score = -1
                best_cand = valid_candidates[0]
                for cand in valid_candidates[:8]:
                    new_diffs = {cand - a for a in row}
                    next_count = 0
                    for next_cand in range(cand + 1, cand + 25):
                        next_diffs = {next_cand - a for a in row}
                        next_diffs.add(next_cand - cand)
                        if not (next_diffs & (used | new_diffs)):
                            next_count += 1
                        if next_count >= 5:
                            break
                    # Prefer smaller values with good future options
                    score = next_count * 100 - cand
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                x = best_cand
                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # ---- Constructor: randomized greedy with flexibility bias ----
    def randomized_greedy(n_rows, k, rng):
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                window = max(30, 80 - 10 * row_idx)
                valid_candidates = []
                for candidate in range(x + 1, x + window + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= 15:
                            break
                if not valid_candidates:
                    candidate = x + 1
                    while True:
                        new_diffs = {candidate - a for a in row}
                        if not new_diffs & used:
                            valid_candidates.append(candidate)
                            break
                        candidate += 1

                # Bias toward smaller values, but occasionally take a flexible one
                if rng.random() < 0.6:
                    x = valid_candidates[0]
                else:
                    # Pick among the first few by a flexibility-weighted choice
                    pool = valid_candidates[:min(6, len(valid_candidates))]
                    x = rng.choice(pool)

                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # ---- Local search: multi-element reduction ----
    def local_search(rows, k, max_iters=300):
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        improved_any = False
        for _ in range(max_iters):
            if _time_left() < 2.0:
                break
            # Work on the row with the largest last entry
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_idx]
            found = False

            # Try to reduce the last element
            old_diffs = {row[-1] - a for a in row[:-1]}
            for new_val in range(row[-2] + 1, row[-1]):
                new_diffs = {new_val - a for a in row[:-1]}
                if new_diffs & (used - old_diffs):
                    continue
                used -= old_diffs
                used |= new_diffs
                row[-1] = new_val
                found = True
                improved_any = True
                break

            if not found:
                # Try to reduce an interior element (index 1..k-1) to shrink the row
                # We attempt to lower a single element while keeping the row increasing
                # and all differences distinct.
                interior_moved = False
                for pos in range(len(row) - 2, 0, -1):  # from near the end toward index 1
                    cur = row[pos]
                    # Candidate new values: between row[pos-1]+1 and cur-1
                    for new_val in range(cur - 1, row[pos - 1], -1):
                        if new_val <= row[pos - 1]:
                            break
                        # Ensure new_val < row[pos+1] if it exists (keep increasing)
                        if pos + 1 < len(row) and new_val >= row[pos + 1]:
                            continue
                        # Recompute diffs involving position pos
                        old_d = set()
                        for j in range(len(row)):
                            if j == pos:
                                continue
                            if j < pos:
                                old_d.add(cur - row[j])
                            else:
                                old_d.add(row[j] - cur)
                        new_d = set()
                        for j in range(len(row)):
                            if j == pos:
                                continue
                            if j < pos:
                                new_d.add(new_val - row[j])
                            else:
                                new_d.add(row[j] - new_val)
                        if new_d & (used - old_d):
                            continue
                        # Apply
                        used -= old_d
                        used |= new_d
                        row[pos] = new_val
                        interior_moved = True
                        improved_any = True
                        found = True
                        break
                    if found:
                        break

            if not found:
                # No improvement possible on this row in this pass; break to avoid loop
                break

        if improved_any:
            return rows
        return None

    # ---- Validation ----
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    rng = random.Random(42)

    # Baseline: deterministic greedy
    baseline = greedy_construct(n_rows, k)
    best_rows = baseline
    best_max = max(max(r) for r in baseline)

    # Multi-strategy restarts
    num_attempts = 60
    for attempt in range(num_attempts):
        if _time_left() < 5.0:
            break
        if attempt % 3 == 0:
            rows = lookahead_greedy(n_rows, k, rng)
        else:
            rows = randomized_greedy(n_rows, k, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            # Refine with local search
            if _time_left() > 5.0:
                improved = local_search(best_rows, k, max_iters=150)
                if improved:
                    new_max = max(max(r) for r in improved)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = improved

    # Final aggressive local search
    if _time_left() > 3.0:
        improved = local_search(best_rows, k, max_iters=600)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)

    return best_rows
