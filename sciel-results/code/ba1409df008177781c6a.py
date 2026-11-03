"""
Difference triangle set (7,5) – global local-search / constraint-satisfaction.

Structural change vs. the seed: instead of committing rows one-by-one with
greedy + per-row backtracking (which locks in a poor global structure), we
treat the whole 7x6 array as one object and run a *global* simulated-annealing
local search over it. Differences are tracked with a Python-integer bitset for
O(1) collision tests, so we can afford thousands of moves. The objective is the
scope (max entry) with a small penalty for "slack", and moves reconstruct random
rows under a global bound so the search can shrink the maximum across rows.

We combine:
  1. a fast greedy baseline (always valid),
  2. iterative deepening on the scope bound M to find the smallest feasible M,
  3. a scope-biased simulated-annealing refinement that lowers the maximum.
A deadline guard returns the best valid result before the wall-clock limit.
"""
import time
import random
import math

# EVOLVE-BLOCK-START


def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
    used = 0  # bitset of used differences
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= 1 << (x - a)
            if new & used:
                continue
            row.append(x)
            used |= new
        rows.append(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


def _validate(rows, k):
    """Check that all rows are valid and all differences are distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        prev = -1
        for i in range(len(row)):
            if row[i] <= prev:
                return False
            prev = row[i]
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in used:
                    return False
                used.add(d)
    return True


def _row_diff_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            m |= 1 << (row[j] - row[i])
    return m


def _build_row(used, M, k, rng, prefer_small=True):
    """
    Build one row of k+1 entries starting at 0, all <= M, with no difference
    colliding with `used` (a bitset). Returns (row, mask) or (None, None).
    Uses a randomized greedy with a small lookahead to avoid dead-ends.
    """
    target = k + 1
    row = [0]
    # We will grow the row; keep a running mask of this row's own diffs.
    row_mask = 0

    def extend(row, row_mask, depth):
        if depth == target:
            return row, row_mask
        last = row[-1]
        start = last + 1
        end = M
        if end < start:
            return None, None
        # candidate ordering
        if prefer_small:
            order = list(range(start, end + 1))
        else:
            order = list(range(start, end + 1))
            rng.shuffle(order)
        # With some probability, try a few random candidates first.
        if prefer_small and len(order) > 8 and rng.random() < 0.5:
            head = rng.sample(order, min(4, len(order)))
            rest = [c for c in order if c not in set(head)]
            order = head + rest

        for cand in order:
            add = 0
            for a in row:
                add |= 1 << (cand - a)
            if add & used or add & row_mask:
                continue
            row.append(cand)
            res_row, res_mask = extend(row, row_mask | add, depth + 1)
            if res_row is not None:
                return res_row, res_mask
            row.pop()
        return None, None

    return extend(row, row_mask, 1)


def _construct_with_bound(n_rows, k, M, rng, deadline, time_slice=1.5):
    """
    Build n_rows rows all <= M with globally distinct differences.
    Returns list of rows on success, None on failure.
    """
    used = 0
    rows = []
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        # a few attempts with different randomizations
        row = None
        for attempt in range(6):
            if time.time() > deadline:
                return None
            prefer = (attempt % 3) != 2
            row, mask = _build_row(used, M, k, rng, prefer_small=prefer)
            if row is not None:
                break
        if row is None:
            return None
        rows.append(row)
        used |= _row_diff_mask(row)
    return rows


def _iterative_deepening(n_rows, k, start_M, max_M, rng, deadline):
    """Try M from start_M..max_M; return (rows, M) or (None, None)."""
    for M in range(start_M, max_M + 1):
        if time.time() > deadline:
            return None, None
        # a handful of randomized attempts per M
        for _ in range(4):
            if time.time() > deadline:
                break
            res = _construct_with_bound(n_rows, k, M, rng, deadline)
            if res is not None and _validate(res, k):
                return res, M
    return None, None


def _sa_refine(rows, n_rows, k, rng, deadline):
    """
    Global simulated-annealing local search to reduce the scope.
    State: list of rows. Objective: scope (max entry).
    Move: pick a random row, rebuild it under a bound = current scope (or a
          slightly smaller target), accepting if the new scope is no larger
          (or with a small probability when larger, to escape).
    Returns the best valid rows seen.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    cur_rows = [list(r) for r in rows]
    cur_scope = _scope(cur_rows)

    # current temperature
    T = max(2.0, 0.05 * best_scope)
    T_min = 0.05
    cool = 0.9995

    def recompute_scope(rs):
        return max(max(r) for r in rs)

    steps = 0
    while time.time() < deadline:
        steps += 1
        # pick a random row to rebuild
        idx = rng.randrange(n_rows)
        # bound for the new row: try current scope, occasionally smaller
        if rng.random() < 0.6:
            M = cur_scope
        else:
            M = max(k, cur_scope - rng.randrange(1, 4))
        # rebuild the row with all other rows' differences fixed
        other_mask = 0
        for j, r in enumerate(cur_rows):
            if j != idx:
                other_mask |= _row_diff_mask(r)
        new_row, _ = _build_row(other_mask, M, k, rng,
                                prefer_small=(rng.random() < 0.7))
        if new_row is None:
            # occasionally just perturb by trying a smaller bound
            continue
        # compute new scope
        trial = list(cur_rows)
        trial[idx] = new_row
        new_scope = recompute_scope(trial)
        delta = new_scope - cur_scope
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            cur_rows = trial
            cur_scope = new_scope
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [list(r) for r in cur_rows]
        T *= cool
        if T < T_min:
            # gentle restart of temperature if no improvement for a while
            T = max(T_min, 0.5 * T)
    return best_rows, best_scope


def construct(n_rows, k):
    t0 = time.time()
    total_budget = 140.0  # safety margin before the 150s kill
    deadline = t0 + total_budget

    # Step 1: greedy baseline (always valid)
    best_rows = _greedy_fallback(n_rows, k)
    best_M = _scope(best_rows)

    rng = random.Random(20250711)

    # Step 2: iterative deepening on the scope bound
    # counting bound: M(M+1)/2 >= n_rows * k*(k+1)/2
    num_diffs = n_rows * k * (k + 1) // 2
    lb = math.ceil((-1 + math.sqrt(1 + 8 * num_diffs)) / 2)
    lb = max(lb, k)

    search_start = lb
    # search a modest window above the bound up to just below the greedy scope
    search_end = min(best_M - 1, lb + 60)
    if search_start <= search_end:
        res, found_M = _iterative_deepening(
            n_rows, k, search_start, search_end, rng, deadline
        )
        if res is not None and found_M < best_M:
            best_rows = res
            best_M = found_M

    # Step 3: global simulated-annealing refinement to push the scope down
    if time.time() < deadline - 8.0:
        refined, refined_scope = _sa_refine(
            best_rows, n_rows, k, rng, deadline - 4.0
        )
        if refined_scope < best_M:
            best_rows = refined
            best_M = refined_scope

    # Final validation; fall back to greedy if anything went wrong
    if not _validate(best_rows, k):
        best_rows = _greedy_fallback(n_rows, k)

    return best_rows

# EVOLVE-BLOCK-END
