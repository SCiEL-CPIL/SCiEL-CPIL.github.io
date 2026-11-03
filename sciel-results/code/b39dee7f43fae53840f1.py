"""Difference triangle set (4,7) — combinatorial row-packing via exhaustive enumeration."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START


def _row_diffs(row):
    """Return frozenset of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _enumerate_valid_rows(k, M):
    """
    Enumerate all valid rows: strictly increasing sequences of length k+1
    starting at 0, max <= M, with all internal differences distinct.
    Returns list of (row_tuple, diff_frozenset).
    """
    valid_rows = []
    # Choose k values from {1, 2, ..., M}
    # Row is (0, a1, a2, ..., ak) where 0 < a1 < a2 < ... < ak <= M
    for combo in combinations(range(1, M + 1), k):
        row = (0,) + combo
        # Check all differences are distinct
        diffs = set()
        n = len(row)
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if not ok:
                break
        if ok:
            valid_rows.append((row, frozenset(diffs)))
    return valid_rows


def _greedy_pack(rows_catalog, n_rows, rng):
    """
    Greedily select n_rows non-conflicting rows from the catalog.
    Uses a heuristic: prefer rows that use "rare" differences.
    Returns the selected rows (as lists) or None if packing fails.
    """
    if not rows_catalog:
        return None

    # Count frequency of each difference across all rows
    diff_freq = {}
    for row, diffs in rows_catalog:
        for d in diffs:
            diff_freq[d] = diff_freq.get(d, 0) + 1

    # Score each row: sum of inverse frequencies (rarer diffs = better)
    # We want to pick rows that use differences that few other rows use
    scored_rows = []
    for idx, (row, diffs) in enumerate(rows_catalog):
        score = 0.0
        for d in diffs:
            freq = diff_freq.get(d, 1)
            score += 1.0 / freq
        scored_rows.append((score, idx, row, diffs))

    # Sort by score descending (prefer rows with rarer differences)
    # Add some randomness for diversity
    scored_rows.sort(key=lambda x: -x[0])

    selected = []
    used_diffs = set()

    for score, idx, row, diffs in scored_rows:
        if len(selected) == n_rows:
            break
        if diffs & used_diffs:
            continue
        selected.append(row)
        used_diffs |= diffs

    if len(selected) == n_rows:
        return [list(r) for r in selected]
    return None


def _greedy_pack_randomized(rows_catalog, n_rows, rng, n_trials=200):
    """
    Multiple randomized greedy packing attempts.
    """
    if not rows_catalog:
        return None

    diff_freq = {}
    for row, diffs in rows_catalog:
        for d in diffs:
            diff_freq[d] = diff_freq.get(d, 0) + 1

    best_result = None

    for trial in range(n_trials):
        # Shuffle with weighted probability based on rarity
        weighted_rows = []
        for idx, (row, diffs) in enumerate(rows_catalog):
            weight = 0.0
            for d in diffs:
                freq = diff_freq.get(d, 1)
                weight += 1.0 / freq
            # Add noise
            weight *= (0.5 + rng.random())
            weighted_rows.append((weight, idx, row, diffs))

        # Partial shuffle: pick top candidates randomly
        rng.shuffle(weighted_rows)
        weighted_rows.sort(key=lambda x: -x[0])

        # Take top 3x n_rows candidates
        candidates = weighted_rows[:min(len(weighted_rows), n_rows * 3)]

        selected = []
        used_diffs = set()

        for weight, idx, row, diffs in candidates:
            if len(selected) == n_rows:
                break
            if diffs & used_diffs:
                continue
            selected.append(row)
            used_diffs |= diffs

        if len(selected) == n_rows:
            scope = max(max(r) for r in selected)
            if best_result is None or scope < max(max(r) for r in best_result):
                best_result = [list(r) for r in selected]

    return best_result


def _local_search_improve(rows, k, deadline, rng):
    """
    Try to reduce the scope by rebuilding individual rows with smaller max.
    """
    if not rows:
        return rows

    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)
    current_scope = best_scope

    while time.time() < deadline:
        improved = False

        for row_idx in range(len(best)):
            # Compute used diffs from other rows
            used_others = set()
            for i, r in enumerate(best):
                if i != row_idx:
                    used_others |= _row_diffs(r)

            # Try to rebuild this row with smaller max
            old_max = max(best[row_idx])

            # Try scopes from 1 to old_max - 1
            for target_max in range(1, old_max):
                # Enumerate valid rows with max <= target_max
                # But we need exactly k+1 elements starting at 0
                # So max must be at least k (since we need k positive values)
                if target_max < k:
                    break

                # Quick check: can we fit k values in [1, target_max]?
                # We need k distinct values from 1..target_max, so target_max >= k
                if target_max < k:
                    continue

                # Try greedy rebuild
                new_row = [0]
                x = 0
                while len(new_row) < k + 1 and x < target_max:
                    x += 1
                    new_diffs = {x - a for a in new_row}
                    if not (new_diffs & used_others):
                        new_row.append(x)

                if len(new_row) == k + 1 and max(new_row) <= target_max:
                    # Verify full validity
                    trial = [list(r) for r in best]
                    trial[row_idx] = new_row
                    # Check all diffs are distinct
                    all_diffs = set()
                    valid = True
                    for r in trial:
                        rd = _row_diffs(r)
                        if rd & all_diffs:
                            valid = False
                            break
                        all_diffs |= rd
                    if valid:
                        new_scope = max(max(r) for r in trial)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            best = trial
                            if new_scope < best_scope:
                                best_scope = new_scope
                            improved = True
                            break  # Move to next row

        if not improved and current_scope == best_scope:
            # No improvement in this pass, do some randomized perturbation
            row_idx = rng.randrange(len(best))
            used_others = set()
            for i, r in enumerate(best):
                if i != row_idx:
                    used_others |= _row_diffs(r)

            # Randomized rebuild
            new_row = [0]
            x = 0
            max_val = max(best[row_idx])
            while len(new_row) < k + 1 and x < max_val + 10:
                x += 1
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & used_others) and rng.random() < 0.8:
                    new_row.append(x)

            if len(new_row) == k + 1:
                trial = [list(r) for r in best]
                trial[row_idx] = new_row
                all_diffs = set()
                valid = True
                for r in trial:
                    rd = _row_diffs(r)
                    if rd & all_diffs:
                        valid = False
                        break
                    all_diffs |= rd
                if valid:
                    sc = max(max(r) for r in trial)
                    if sc <= current_scope:
                        best = trial
                        current_scope = sc
                        if sc < best_scope:
                            best_scope = sc

    return best


def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    rng = random.Random(12345)

    # Counting bound: n_rows * k*(k+1)/2 <= M*(M+1)/2
    # For n_rows=4, k=7: 4*28 = 112, so M*(M+1)/2 >= 112 => M >= 15
    # But we need each row to have k+1=8 elements, so M must be >= k = 7
    # Practical lower bound: max(k, ceil((-1+sqrt(1+8*n_rows*k*(k+1)/2))/2))
    import math
    total_diffs = n_rows * k * (k + 1) // 2
    M_low = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    M_low = max(M_low, k)

    M_high = 200  # Upper bound for binary search

    best_result = None
    best_scope = float('inf')

    # First, try a quick greedy to get an initial upper bound
    quick_rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        quick_rows.append(row)
    quick_scope = max(max(r) for r in quick_rows)
    best_result = [list(r) for r in quick_rows]
    best_scope = quick_scope
    M_high = min(M_high, quick_scope)

    # Binary search on M
    lo = M_low
    hi = M_high

    while lo <= hi and time.time() < deadline - 10.0:
        mid = (lo + hi) // 2

        # Enumerate all valid rows with max <= mid
        catalog = _enumerate_valid_rows(k, mid)

        if not catalog:
            lo = mid + 1
            continue

        # Try to pack n_rows non-conflicting rows
        result = _greedy_pack(catalog, n_rows, rng)

        if result is not None:
            scope = max(max(r) for r in result)
            if scope < best_scope:
                best_scope = scope
                best_result = result
            hi = mid - 1
        else:
            # Try randomized packing
            result = _greedy_pack_randomized(catalog, n_rows, rng, n_trials=100)
            if result is not None:
                scope = max(max(r) for r in result)
                if scope < best_scope:
                    best_scope = scope
                    best_result = result
                hi = mid - 1
            else:
                lo = mid + 1

    # Local search to improve the best result found
    if best_result is not None and time.time() < deadline - 5.0:
        best_result = _local_search_improve(best_result, k, deadline - 3.0, rng)
        best_scope = max(max(r) for r in best_result)

    # If we still have time, try more randomized packing at the best scope
    if best_result is not None and time.time() < deadline - 5.0:
        current_scope = max(max(r) for r in best_result)
        # Try to find a packing at current_scope or lower
        for target_M in range(max(M_low, current_scope - 5), current_scope + 1):
            if time.time() > deadline - 3.0:
                break
            catalog = _enumerate_valid_rows(k, target_M)
            if not catalog:
                continue
            result = _greedy_pack_randomized(catalog, n_rows, rng, n_trials=500)
            if result is not None:
                scope = max(max(r) for r in result)
                if scope < best_scope:
                    best_scope = scope
                    best_result = result

    return best_result

# EVOLVE-BLOCK-END
