"""Difference triangle set solver: Hybrid SA with row-rebuild + entry-shift moves."""
import time
import sys
import random
import math
from collections import Counter

sys.setrecursionlimit(100000)


def _validate(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _row_diffs(row):
    n = len(row)
    return [row[j] - row[i] for i in range(n) for j in range(i + 1, n)]


def _greedy(n_rows, k):
    """Fast greedy: smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows


def _build_row_given_used(used, k, rng, cap=None, jitter=0.0):
    """Build a row of k+1 numbers starting at 0, avoiding used differences.
    Returns None if cannot complete within cap."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if cap is not None and x > cap:
            return None
        if jitter > 0 and rng.random() < jitter:
            x += rng.randint(1, 3)
            if cap is not None and x > cap:
                return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used = used | new_diffs
    return row


def _initial_construction(n_rows, k, rng, max_attempts=200):
    """Build initial valid set with varied diversity parameters."""
    best = None
    best_score = float('inf')
    start = time.time()
    
    for attempt in range(max_attempts):
        if time.time() - start > 15.0:
            break
        used = set()
        rows = []
        ok = True
        for r in range(n_rows):
            div = 0.04 * (0.5 + rng.random())
            row = _build_row_given_used(used, k, rng, jitter=div)
            if row is None:
                ok = False
                break
            rows.append(row)
            used = used | set(_row_diffs(row))
        
        if ok and len(rows) == n_rows:
            if _validate(rows, k):
                score = _scope(rows)
                if score < best_score:
                    best_score = score
                    best = [list(r) for r in rows]
    
    if best is None:
        best = _greedy(n_rows, k)
        best_score = _scope(best)
    return best, best_score


def _sa_optimize(n_rows, k, initial_rows, deadline, seed=42):
    """Simulated Annealing with row-rebuild and entry-shift moves."""
    rng = random.Random(seed)
    rows = [list(r) for r in initial_rows]
    L = k + 1  # entries per row

    # Initialize diff counts
    row_diffs = [_row_diffs(r) for r in rows]
    diff_counts = Counter()
    for rd in row_diffs:
        for d in rd:
            diff_counts[d] += 1

    def compute_conflicts():
        return sum(max(0, c - 1) for c in diff_counts.values())

    def compute_cost():
        conflicts = compute_conflicts()
        max_val = max(r[-1] for r in rows)
        return max_val + 1000 * conflicts

    current_cost = compute_cost()
    best_cost = current_cost
    best_rows = [r[:] for r in rows]
    best_max = max(r[-1] for r in best_rows)

    T = 20.0
    T_min = 0.05
    cooling = 0.9998

    while time.time() < deadline:
        if T < T_min:
            break

        r_idx = rng.randrange(n_rows)
        move_type = rng.random()

        if move_type < 0.60:
            # Row rebuild: rebuild row r_idx against other rows' diffs
            old_row = rows[r_idx]
            old_diffs = row_diffs[r_idx]
            
            # Remove old diffs
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Build used set from remaining diffs
            used = set(diff_counts.keys())
            
            # Try with progressively tighter caps
            base_cap = best_max
            new_row = None
            for attempt in range(3):
                cap = base_cap - attempt * 2
                new_row = _build_row_given_used(used, k, rng, cap=cap, jitter=0.05)
                if new_row is not None:
                    break
            
            if new_row is None:
                # Restore old row
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                T *= cooling
                continue
            
            new_diffs = _row_diffs(new_row)
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = compute_cost()
            delta = new_cost - current_cost

            if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
                rows[r_idx] = new_row
                row_diffs[r_idx] = new_diffs
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [r[:] for r in rows]
                    best_max = max(r[-1] for r in best_rows)
            else:
                # Revert
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                rows[r_idx] = old_row
                row_diffs[r_idx] = old_diffs

        elif move_type < 0.85:
            # Entry shift: change one entry in a row
            i_idx = rng.randrange(1, L)
            row = rows[r_idx]
            old_val = row[i_idx]

            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else best_max + 20
            upper = min(upper, best_max + 30)

            if lower > upper:
                T *= cooling
                continue

            # Bias toward reducing
            if rng.random() < 0.6:
                new_val = rng.randint(lower, max(lower, old_val - 1))
            else:
                new_val = rng.randint(lower, upper)

            if new_val == old_val:
                T *= cooling
                continue

            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]

            row[i_idx] = new_val
            new_diffs = _row_diffs(row)
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1

            new_cost = compute_cost()
            delta = new_cost - current_cost

            if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
                row_diffs[r_idx] = new_diffs
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [r[:] for r in rows]
                    best_max = max(r[-1] for r in best_rows)
            else:
                row[i_idx] = old_val
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                row_diffs[r_idx] = old_diffs

        else:
            # Row swap: swap two rows to explore permutation space
            s_idx = rng.randrange(n_rows)
            if s_idx == r_idx:
                T *= cooling
                continue
            rows[r_idx], rows[s_idx] = rows[s_idx], rows[r_idx]
            row_diffs[r_idx], row_diffs[s_idx] = row_diffs[s_idx], row_diffs[r_idx]
            # Cost doesn't change with row swap, so just accept
            # (keeps diversity in search)

        T *= cooling

    return best_rows


def _hill_climb_rows(rows, time_limit, row_len, n_rows):
    """Iteratively reconstruct rows to shrink the bounding box."""
    rows = [r[:] for r in rows]
    max_val = max(max(r) for r in rows)
    start_time = time.time()
    rng = random.Random(777)

    # Build used_diffs set
    used_diffs = set()
    for r in rows:
        for d in _row_diffs(r):
            used_diffs.add(d)

    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for r_idx in range(n_rows):
            if time.time() - start_time >= time_limit:
                break
            old_max = rows[r_idx][-1]
            
            # Remove current row's diffs
            current_diffs = set(_row_diffs(rows[r_idx]))
            for d in current_diffs:
                used_diffs.discard(d)
            
            # Try to rebuild with tight bound
            new_row = _build_row_given_used(used_diffs, k=len(rows[r_idx]) - 2, rng=rng, cap=max_val, jitter=0.0)
            
            if new_row is not None and len(new_row) == row_len:
                new_diffs = set(_row_diffs(new_row))
                if not (new_diffs & used_diffs):
                    rows[r_idx] = new_row
                    used_diffs |= new_diffs
                    new_max = rows[r_idx][-1]
                    if new_max < old_max:
                        improved = True
                        new_global = max(max(r) for r in rows)
                        if new_global < max_val:
                            max_val = new_global
                else:
                    used_diffs |= current_diffs
            else:
                used_diffs |= current_diffs

    return rows, max(max(r) for r in rows)


def construct(n_rows, k):
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s
    row_len = k + 1
    rng = random.Random(12345)

    # Phase 1: Initial construction with diversity
    best, best_scope = _initial_construction(n_rows, k, rng, max_attempts=300)

    # Phase 2: Hill climbing with row reconstruction
    if time.time() < deadline - 40.0:
        try:
            hc_time = min(20.0, deadline - time.time() - 30.0)
            hc_rows, hc_scope = _hill_climb_rows(best, hc_time, row_len, n_rows)
            if _validate(hc_rows, k) and hc_scope < best_scope:
                best = hc_rows
                best_scope = hc_scope
        except Exception:
            pass

    # Phase 3: Main SA with row-rebuild moves
    if time.time() < deadline - 25.0:
        try:
            sa_deadline = min(deadline - 10.0, time.time() + 50.0)
            sa_rows = _sa_optimize(n_rows, k, best, sa_deadline, seed=42)
            if sa_rows is not None and _validate(sa_rows, k):
                sa_scope = _scope(sa_rows)
                if sa_scope < best_scope:
                    best = sa_rows
                    best_scope = sa_scope
        except Exception:
            pass

    # Phase 4: Second SA pass with different seed for diversity
    if time.time() < deadline - 15.0:
        try:
            sa2_deadline = min(deadline - 8.0, time.time() + 30.0)
            sa2_rows = _sa_optimize(n_rows, k, best, sa2_deadline, seed=999)
            if sa2_rows is not None and _validate(sa2_rows, k):
                sa2_scope = _scope(sa2_rows)
                if sa2_scope < best_scope:
                    best = sa2_rows
                    best_scope = sa2_scope
        except Exception:
            pass

    # Phase 5: Final hill-climb polish
    if time.time() < deadline - 8.0:
        try:
            hc2_time = min(10.0, deadline - time.time() - 5.0)
            hc2_rows, hc2_scope = _hill_climb_rows(best, hc2_time, row_len, n_rows)
            if _validate(hc2_rows, k) and hc2_scope < best_scope:
                best = hc2_rows
                best_scope = hc2_scope
        except Exception:
            pass

    # Final validation
    if not _validate(best, k):
        best = _greedy(n_rows, k)

    return best
