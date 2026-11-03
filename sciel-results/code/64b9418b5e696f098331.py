"""
Combined difference-triangle solver.

Strategy:
  Phase 1 – Fast randomised greedy with look-ahead (many restarts).
  Phase 2 – Simulated-annealing local search on individual entries.
  Phase 3 – DFS row re-optimisation: for each row (in size order), try to
            minimise its max entry given the other rows are fixed.  This
            escapes greedy basins and pushes scope toward the best-known value.
  Phase 4 – Final polish pass of the same re-optimisation.

All phases are bounded by an explicit wall-clock deadline (140 s) so the
program always returns the best result found.
"""

import time
import random
import sys

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _greedy_fallback(n_rows, k):
    """Deterministic greedy: always valid, used as safety net."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


def _all_diffs(rows):
    """Return the set of all positive pairwise differences across all rows."""
    s = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(r[j] - r[i])
    return s


def _validate(rows):
    diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True


# ---------------------------------------------------------------------------
# Phase 1 – Randomised greedy with look-ahead
# ---------------------------------------------------------------------------

def _randomized_greedy(n_rows, k, rng, look_ahead=6):
    used, rows = set(), []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            start = row[-1] + 1
            cands = []
            for cand in range(start, start + look_ahead + 1):
                nd = {cand - a for a in row}
                if not nd & used:
                    cands.append(cand)
            if not cands:
                cand = start
                while True:
                    nd = {cand - a for a in row}
                    if not nd & used:
                        cands.append(cand)
                        break
                    cand += 1
                    if cand > 500:
                        break
            if not cands:
                # absolute fallback
                cand = start
                while True:
                    nd = {cand - a for a in row}
                    if not nd & used:
                        cands.append(cand)
                        break
                    cand += 1
            cands.sort()
            if rng.random() < 0.8 or len(cands) == 1:
                chosen = cands[0]
            else:
                chosen = cands[rng.randrange(min(3, len(cands)))]
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Phase 2 – Simulated-annealing local search
# ---------------------------------------------------------------------------

def _anneal_local_search(rows, k, deadline, rng):
    """
    Simulated annealing on individual entries.
    A 'move' picks a random row and position (1..k) and tries a new value
    in [row[p-1]+1, row[p]+delta] (or down to row[p-1]+1).
    Accept if scope decreases, or with probability exp(-dE/T).
    """
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    cur = [list(r) for r in rows]
    cur_sc = best_sc

    T = 30.0
    T_min = 0.5
    cooling = 0.995
    delta = 3

    t0 = time.time()
    while time.time() < deadline:
        if T < T_min:
            # restart annealing with a small perturbation
            T = 10.0
            # perturb: random entry in a random row
            ri = rng.randrange(len(cur))
            p = rng.randrange(1, len(cur[ri]))
            lo = cur[ri][p - 1] + 1
            hi = cur[ri][p] + 5
            new_val = rng.randint(lo, hi)
            cur[ri][p] = new_val
            # recompute scope (may be invalid temporarily; we'll fix)
            if not _validate(cur):
                # revert
                cur[ri][p] = best[ri][p]
                cur_sc = best_sc
            else:
                cur_sc = _scope(cur)
                if cur_sc < best_sc:
                    best_sc = cur_sc
                    best = [list(r) for r in cur]
            continue

        # pick a move
        ri = rng.randrange(len(cur))
        p = rng.randrange(1, len(cur[ri]))
        lo = cur[ri][p - 1] + 1
        hi = cur[ri][p] + delta
        new_val = rng.randint(lo, hi)
        if new_val == cur[ri][p]:
            continue
        old_val = cur[ri][p]
        cur[ri][p] = new_val

        # validate quickly
        if not _validate(cur):
            cur[ri][p] = old_val
            T *= cooling
            continue

        new_sc = _scope(cur)
        dE = new_sc - cur_sc
        if dE <= 0 or rng.random() < pow(2.718281828, -dE / T):
            cur_sc = new_sc
            if new_sc < best_sc:
                best_sc = new_sc
                best = [list(r) for r in cur]
        else:
            cur[ri][p] = old_val
        T *= cooling
    return best, best_sc


# ---------------------------------------------------------------------------
# Phase 3 – DFS row re-optimisation
# ---------------------------------------------------------------------------

def _optimize_row(row, k, other_diffs, deadline, rng):
    """
    Try to minimise the max entry of 'row' given that all differences in
    'other_diffs' are already taken.  Returns (improved_row, improved_max).
    Uses a simple backtracking / iterative-deepening approach on the
    entries of the row from right to left.
    """
    n = len(row)  # k+1
    cur_max = row[-1]
    best_row = list(row)
    best_max = cur_max

    def _row_diffs(r):
        s = set()
        for i in range(n):
            for j in range(i + 1, n):
                s.add(r[j] - r[i])
        return s

    def _valid_row(r, od):
        rd = _row_diffs(r)
        if rd & od:
            return False
        # internal duplicates
        if len(rd) != n * (n - 1) // 2:
            return False
        return True

    # Try to reduce max from cur_max downward
    for target_max in range(cur_max - 1, -1, -1):
        if time.time() > deadline:
            break
        # Can we find a row with max <= target_max that is valid?
        # Build greedily with backtracking, trying to keep entries small.
        found = _dfs_build_row(n, target_max, other_diffs, deadline, rng)
        if found is not None:
            return found, found[-1]
    return best_row, best_max


def _dfs_build_row(n, max_allowed, other_diffs, deadline, rng):
    """
    Build a row of n entries starting at 0, strictly increasing, max <= max_allowed,
    with all internal diffs distinct and disjoint from other_diffs.
    Returns the row or None.
    """
    row = [0]

    def _build(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        lo = row[-1] + 1
        hi = max_allowed
        # Try candidates in randomised order for diversity
        cands = list(range(lo, hi + 1))
        if len(cands) > 15:
            # limit candidates to keep DFS tractable: prefer smaller
            rng.shuffle(cands)
            cands.sort()
            cands = cands[:15]
        for cand in cands:
            row.append(cand)
            # check new diffs
            nd = {cand - a for a in row[:-1]}
            if nd & other_diffs:
                row.pop()
                continue
            # check internal duplicates in full row so far
            ok = True
            seen = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        ok = False
                        break
                    seen.add(d)
                if not ok:
                    break
            if not ok:
                row.pop()
                continue
            if _build(pos + 1):
                return True
            row.pop()
        return False

    if _build(1):
        return list(row)
    return None


def _row_reoptimize(rows, k, deadline, rng):
    """
    Repeatedly try to minimise each row's max entry, in order of current
    max (largest first – they constrain the most).  Multiple passes.
    """
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    n_rows = len(best)

    for _pass in range(4):
        if time.time() > deadline:
            break
        # order rows by current max descending
        order = sorted(range(n_rows), key=lambda i: best[i][-1], reverse=True)
        improved = False
        for ri in order:
            if time.time() > deadline:
                break
            # compute other_diffs
            other_diffs = set()
            for rj in range(n_rows):
                if rj == ri:
                    continue
                r = best[rj]
                n = len(r)
                for i in range(n):
                    for j in range(i + 1, n):
                        other_diffs.add(r[j] - r[i])
            new_row, new_max = _optimize_row(best[ri], k, other_diffs, deadline, rng)
            if new_max < best[ri][-1]:
                best[ri] = new_row
                improved = True
                best_sc = _scope(best)
    return best, best_sc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 140.0  # 10 s safety margin before 150 s kill
    rng = random.Random(42)

    # Phase 1: many randomised greedy restarts
    best_rows = _greedy_fallback(n_rows, k)
    best_sc = _scope(best_rows)

    phase1_deadline = t_start + 20.0
    while time.time() < phase1_deadline:
        rows = _randomized_greedy(n_rows, k, rng, look_ahead=rng.choice([4, 5, 6, 8]))
        sc = _scope(rows)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows

    # Phase 2: simulated-annealing local search
    phase2_deadline = t_start + 60.0
    best_rows, best_sc = _anneal_local_search(best_rows, k, phase2_deadline, rng)

    # Phase 3: DFS row re-optimisation
    phase3_deadline = t_start + 130.0
    best_rows, best_sc = _row_reoptimize(best_rows, k, phase3_deadline, rng)

    # Phase 4: final polish – one more round of re-optimisation
    phase4_deadline = deadline
    best_rows, best_sc = _row_reoptimize(best_rows, k, phase4_deadline, rng)

    # Final validation
    if not _validate(best_rows):
        # fall back to deterministic greedy
        best_rows = _greedy_fallback(n_rows, k)

    return best_rows

# EVOLVE-BLOCK-END
