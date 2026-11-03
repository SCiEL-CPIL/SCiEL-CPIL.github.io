"""
Combined difference-triangle solver – structural rewrite.

New structural idea (from the high-scoring inspiration):
  * Represent each row by its *consecutive* differences (a Sidon set of
    k positive integers whose partial sums are the row entries).
  * Primary optimisation objective is the scope (max entry); secondary
    objective is the sum of all entries (packing density).
  * Use a bitmask (Python int) for the set of all pairwise differences
    of a row, so collision checks are O(1) bit-ops.
  * The main search is simulated annealing over individual consecutive
    differences (delta moves) and occasional full-row rebuilds in
    difference-space, with a fast cost function.
  * A targeted DFS row-rebuild (most-constrained-row heuristic) is used
    as a local improvement operator: pick the row with the largest max
    entry and try to rebuild it with a smaller max via backtracking in
    difference-space.
  * Initialisation uses many randomised greedy restarts in difference-space.

This replaces the previous "raw integer coordinate" local search and the
expensive DFS over absolute coordinates with a representation that makes
each move cheap and the cost function tight.
"""

import time
import random
import math
import sys

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Bitmask helpers
# ---------------------------------------------------------------------------

def _row_mask(row):
    """Return a bitmask of all positive pairwise differences in row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _row_mask_from_diffs(consec_diffs):
    """Given consecutive differences (length k), build the row and its mask."""
    row = [0]
    for d in consec_diffs:
        row.append(row[-1] + d)
    return _row_mask(row), row


# ---------------------------------------------------------------------------
# Scope / cost helpers
# ---------------------------------------------------------------------------

def _scope(rows):
    return max(r[-1] for r in rows)


def _sum_entries(rows):
    return sum(sum(r) for r in rows)


def _validate(rows):
    """Check all positive pairwise differences are distinct across rows."""
    seen = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                d = r[j] - ri
                if d in seen:
                    return False
                seen.add(d)
    return True


def _greedy_fallback(n_rows, k):
    """Deterministic greedy: always valid, used as safety net."""
    used = set()
    rows = []
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


# ---------------------------------------------------------------------------
# Phase 1 – Randomised greedy in difference-space
# ---------------------------------------------------------------------------

def _randomized_greedy_diff(n_rows, k, rng, look_ahead=6):
    """
    Build rows one by one.  For each row, build consecutive differences
    greedily with randomisation.  A consecutive difference d at position
    p is valid if the new pairwise diffs (d plus d + previous diffs) don't
    collide with used diffs or with the row's internal diffs.
    """
    used_mask = 0
    rows = []
    row_len = k + 1

    for _ in range(n_rows):
        # Build row as list of entries, tracking consecutive diffs
        row = [0]
        consec = []
        local_mask = 0  # mask of diffs within this row so far
        while len(row) < row_len:
            start = 1
            cands = []
            # Candidate consecutive diffs: try small values first
            for d in range(start, start + look_ahead + 2):
                # new entry = row[-1] + d
                new_entry = row[-1] + d
                # new pairwise diffs: (new_entry - a) for a in row
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (new_entry - a))
                if (new_mask & used_mask) or (new_mask & local_mask):
                    continue
                cands.append(d)
            if not cands:
                # extend search
                d = start + look_ahead + 2
                while True:
                    new_entry = row[-1] + d
                    new_mask = 0
                    for a in row:
                        new_mask |= (1 << (new_entry - a))
                    if (new_mask & used_mask) or (new_mask & local_mask):
                        d += 1
                        continue
                    cands.append(d)
                    break
            cands.sort()
            if rng.random() < 0.75 or len(cands) == 1:
                chosen_d = cands[0]
            else:
                chosen_d = cands[rng.randrange(min(3, len(cands)))]
            row.append(row[-1] + chosen_d)
            consec.append(chosen_d)
            # update local_mask
            for a in row[:-1]:
                local_mask |= (1 << (row[-1] - a))
        used_mask |= local_mask
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Phase 2 – Simulated annealing in difference-space
# ---------------------------------------------------------------------------

def _sa_difference_space(rows, k, deadline, rng):
    """
    SA where the state is (rows, diff_counts, row_masks).
    Moves:
      1. Perturb one consecutive difference of a random row.
      2. Rebuild one row via randomised greedy (with the other rows fixed).
    Cost = scope + 0.005 * sum_entries + 10000 * collisions.
    """
    n = len(rows)
    cur_rows = [list(r) for r in rows]
    cur_consec = [[r[i + 1] - r[i] for i in range(len(r) - 1)] for r in cur_rows]
    cur_masks = [_row_mask(r) for r in cur_rows]
    cur_scope = _scope(cur_rows)
    cur_sum = _sum_entries(cur_rows)

    # diff_counts: dict diff -> count
    diff_counts = {}
    for r in cur_rows:
        m = _row_mask(r)
        # iterate set bits
        mm = m
        while mm:
            d = (mm & (-mm)).bit_length() - 1
            diff_counts[d] = diff_counts.get(d, 0) + 1
            mm &= mm - 1

    def _collisions(dc):
        return sum(max(0, c - 1) for c in dc.values())

    def _cost(sc, sm, dc):
        return sc + 0.005 * sm + 10000.0 * _collisions(dc)

    best_rows = [list(r) for r in cur_rows]
    best_consec = [c[:] for c in cur_consec]
    best_masks = cur_masks[:]
    best_scope = cur_scope
    best_sum = cur_sum
    best_dc = dict(diff_counts)
    best_cost = _cost(cur_scope, cur_sum, diff_counts)

    T = 5.0
    T_min = 0.05
    cooling = 0.9997

    t0 = time.time()
    while time.time() < deadline:
        if T < T_min:
            T = 5.0
            # small perturbation: random consecutive diff
            ri = rng.randrange(n)
            p = rng.randrange(len(cur_consec[ri]))
            cur_consec[ri][p] = max(1, cur_consec[ri][p] + rng.randint(-2, 2))
            # rebuild row from consec
            row = [0]
            for d in cur_consec[ri]:
                row.append(row[-1] + d)
            cur_rows[ri] = row
            cur_masks[ri] = _row_mask(row)
            # update diff_counts
            old_dc = dict(diff_counts)
            # remove old row diffs
            old_m = best_masks[ri]  # not quite right; use cur before update
            # Actually we need to recompute. Let's just rebuild diff_counts.
            diff_counts = {}
            for r in cur_rows:
                m = _row_mask(r)
                mm = m
                while mm:
                    d = (mm & (-mm)).bit_length() - 1
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    mm &= mm - 1
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)
            cur_cost = _cost(cur_scope, cur_sum, diff_counts)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_scope = cur_scope
                best_sum = cur_sum
                best_rows = [list(r) for r in cur_rows]
                best_consec = [c[:] for c in cur_consec]
                best_masks = cur_masks[:]
                best_dc = dict(diff_counts)
            continue

        r = rng.random()
        if r < 0.70:
            # Perturb one consecutive difference
            ri = rng.randrange(n)
            p = rng.randrange(len(cur_consec[ri]))
            old_d = cur_consec[ri][p]
            new_d = max(1, old_d + rng.randint(-3, 3))
            if new_d == old_d:
                T *= cooling
                continue
            cur_consec[ri][p] = new_d
            # rebuild row
            row = [0]
            for d in cur_consec[ri]:
                row.append(row[-1] + d)
            new_mask = _row_mask(row)

            # quick collision check: new_mask & (used_without_ri)
            used_without = 0
            for j in range(n):
                if j != ri:
                    used_without |= cur_masks[j]
            if new_mask & used_without:
                # collision – reject
                cur_consec[ri][p] = old_d
                T *= cooling
                continue

            # also check internal duplicates: row must have distinct pairwise diffs
            # (already guaranteed by mask having popcount = k(k+1)/2)
            expected_popcount = k * (k + 1) // 2
            if new_mask.bit_count() != expected_popcount:
                cur_consec[ri][p] = old_d
                T *= cooling
                continue

            cur_rows[ri] = row
            cur_masks[ri] = new_mask
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)

            # update diff_counts incrementally: remove old, add new
            # old row mask
            old_row = [0]
            for d in cur_consec[ri]:
                old_row.append(old_row[-1] + d)
            # Wait, we already changed cur_consec. Let's rebuild diff_counts.
            diff_counts = {}
            for rr in cur_rows:
                m = _row_mask(rr)
                mm = m
                while mm:
                    d = (mm & (-mm)).bit_length() - 1
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    mm &= mm - 1

            new_cost = _cost(cur_scope, cur_sum, diff_counts)
            dE = new_cost - _cost(best_scope, best_sum, best_dc)
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, T_min)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = cur_scope
                    best_sum = cur_sum
                    best_rows = [list(r) for r in cur_rows]
                    best_consec = [c[:] for c in cur_consec]
                    best_masks = cur_masks[:]
                    best_dc = dict(diff_counts)
            else:
                # revert
                cur_consec[ri][p] = old_d
                row = [0]
                for d in cur_consec[ri]:
                    row.append(row[-1] + d)
                cur_rows[ri] = row
                cur_masks[ri] = _row_mask(row)
                diff_counts = {}
                for rr in cur_rows:
                    m = _row_mask(rr)
                    mm = m
                    while mm:
                        d = (mm & (-mm)).bit_length() - 1
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                        mm &= mm - 1
        else:
            # Rebuild one row via randomised greedy
            ri = rng.randrange(n)
            used_without = 0
            for j in range(n):
                if j != ri:
                    used_without |= cur_masks[j]

            # randomised greedy rebuild
            row = [0]
            consec = []
            local_mask = 0
            success = True
            for _j in range(k):
                found = False
                start = 1
                if rng.random() < 0.3:
                    start = rng.randint(1, 4)
                for d in range(start, cur_scope + 10):
                    new_entry = row[-1] + d
                    new_mask = 0
                    for a in row:
                        new_mask |= (1 << (new_entry - a))
                    if (new_mask & used_without) or (new_mask & local_mask):
                        continue
                    row.append(new_entry)
                    consec.append(d)
                    for a in row[:-1]:
                        local_mask |= (1 << (row[-1] - a))
                    found = True
                    break
                if not found:
                    success = False
                    break
            if not success:
                T *= cooling
                continue

            new_mask = _row_mask(row)
            if new_mask & used_without:
                T *= cooling
                continue
            if new_mask.bit_count() != k * (k + 1) // 2:
                T *= cooling
                continue

            cur_rows[ri] = row
            cur_consec[ri] = consec
            cur_masks[ri] = new_mask
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)
            diff_counts = {}
            for rr in cur_rows:
                m = _row_mask(rr)
                mm = m
                while mm:
                    d = (mm & (-mm)).bit_length() - 1
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    mm &= mm - 1
            new_cost = _cost(cur_scope, cur_sum, diff_counts)
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = cur_scope
                best_sum = cur_sum
                best_rows = [list(r) for r in cur_rows]
                best_consec = [c[:] for c in cur_consec]
                best_masks = cur_masks[:]
                best_dc = dict(diff_counts)

        T *= cooling

    return best_rows, best_scope


# ---------------------------------------------------------------------------
# Phase 3 – DFS row rebuild (most-constrained-row heuristic)
# ---------------------------------------------------------------------------

def _dfs_rebuild_row(row_len, max_allowed, used_mask, deadline, rng, node_budget=200000):
    """
    Try to build a row of length row_len with max entry <= max_allowed,
    disjoint from used_mask.  Returns (row, mask) or (None, None).
    Uses backtracking on consecutive differences.
    """
    best_row = [None]
    best_max = [max_allowed + 1]
    nodes = [0]

    def dfs(row, local_mask, consec):
        if time.time() > deadline or nodes[0] >= node_budget:
            return
        nodes[0] += 1
        if len(row) == row_len:
            m = row[-1]
            if m < best_max[0]:
                best_max[0] = m
                best_row[0] = row[:]
            return
        # Try consecutive diffs from 1 upward, but cap by remaining budget
        remaining = best_max[0] - 1 - row[-1]
        if remaining <= 0:
            return
        # Try small diffs first; also try some random larger ones
        max_d = min(remaining, 20)
        for d in range(1, max_d + 1):
            new_entry = row[-1] + d
            new_mask = 0
            for a in row:
                new_mask |= (1 << (new_entry - a))
            if (new_mask & used_mask) or (new_mask & local_mask):
                continue
            row.append(new_entry)
            consec.append(d)
            dfs(row, local_mask | new_mask, consec)
            row.pop()
            consec.pop()
            if nodes[0] >= node_budget or time.time() > deadline:
                return

    dfs([0], 0, [])
    if best_row[0] is None:
        return None, None
    return best_row[0], _row_mask(best_row[0])


def _row_reoptimize(rows, k, deadline, rng):
    """
    Repeatedly try to rebuild the row with the largest max entry.
    Multiple passes until no improvement or deadline.
    """
    n = len(rows)
    cur_rows = [list(r) for r in rows]
    cur_masks = [_row_mask(r) for r in cur_rows]
    best_rows = [list(r) for r in cur_rows]
    best_masks = cur_masks[:]
    best_sc = _scope(best_rows)

    for _pass in range(8):
        if time.time() > deadline:
            break
        # Pick row with largest max entry
        ri = max(range(n), key=lambda i: cur_rows[i][-1])
        cur_max = cur_rows[ri][-1]

        # Compute used mask without this row
        used_without = 0
        for j in range(n):
            if j != ri:
                used_without |= cur_masks[j]

        # Try to rebuild with target max = cur_max - 1, then -2, ...
        improved = False
        for target in range(cur_max - 1, max(cur_max - 8, 0), -1):
            if time.time() > deadline:
                break
            new_row, new_mask = _dfs_rebuild_row(k + 1, target, used_without, deadline, rng, node_budget=100000)
            if new_row is not None:
                cur_rows[ri] = new_row
                cur_masks[ri] = new_mask
                improved = True
                new_sc = _scope(cur_rows)
                if new_sc < best_sc:
                    best_sc = new_sc
                    best_rows = [list(r) for r in cur_rows]
                    best_masks = cur_masks[:]
                break
        if not improved:
            # Try a few more randomised rebuilds
            for _attempt in range(3):
                if time.time() > deadline:
                    break
                # randomised greedy rebuild
                row = [0]
                local_mask = 0
                success = True
                for _j in range(k):
                    found = False
                    for d in range(1, cur_max + 5):
                        new_entry = row[-1] + d
                        new_mask = 0
                        for a in row:
                            new_mask |= (1 << (new_entry - a))
                        if (new_mask & used_without) or (new_mask & local_mask):
                            continue
                        row.append(new_entry)
                        for a in row[:-1]:
                            local_mask |= (1 << (row[-1] - a))
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if success:
                    nm = _row_mask(row)
                    if not (nm & used_without) and nm.bit_count() == k * (k + 1) // 2:
                        cur_rows[ri] = row
                        cur_masks[ri] = nm
                        new_sc = _scope(cur_rows)
                        if new_sc < best_sc:
                            best_sc = new_sc
                            best_rows = [list(r) for r in cur_rows]
                            best_masks = cur_masks[:]
                        break

    return best_rows, best_sc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150 s kill
    rng = random.Random(42)

    # Phase 1: many randomised greedy restarts in difference-space
    best_rows = _greedy_fallback(n_rows, k)
    best_sc = _scope(best_rows)

    phase1_deadline = t_start + 15.0
    while time.time() < phase1_deadline:
        rows = _randomized_greedy_diff(n_rows, k, rng, look_ahead=rng.choice([4, 5, 6, 8]))
        sc = _scope(rows)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows

    # Phase 2: SA in difference-space
    phase2_deadline = t_start + 80.0
    best_rows, best_sc = _sa_difference_space(best_rows, k, phase2_deadline, rng)

    # Phase 3: DFS row re-optimisation
    phase3_deadline = t_start + 125.0
    best_rows, best_sc = _row_reoptimize(best_rows, k, phase3_deadline, rng)

    # Final validation
    if not _validate(best_rows):
        best_rows = _greedy_fallback(n_rows, k)

    return best_rows

# EVOLVE-BLOCK-END
