import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct a difference triangle set for n_rows rows of k+1 integers.
    Returns a list of rows.
    
    Strategy: Enumerate a pool of candidate rows (each a valid Sidon-like set
    with distinct internal differences), then solve an assignment problem to
    pick n_rows rows with pairwise disjoint difference sets, minimizing the
    maximum entry (scope). Uses a greedy + local search hybrid with a
    time-bounded enumeration.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Step 1: Generate initial valid solution using greedy
    initial_rows = _greedy_construct(n_rows, k)
    best_score = _get_scope(initial_rows)
    best_rows = [row[:] for row in initial_rows]

    # Step 2: Enumerate candidate rows with max entry <= some bound
    # Start with a bound slightly above the greedy solution's max
    greedy_max = best_score
    # We'll enumerate candidates with max up to greedy_max (and maybe a bit below)
    # to find better assignments.

    # Enumerate candidate rows: all valid sets {0, a1, ..., ak} with ak <= M
    # where M is a bound. For each, compute its difference set.
    # We want to find n_rows candidates with pairwise disjoint diff sets,
    # minimizing the max entry.

    # To keep this tractable, we'll use a randomized enumeration with a cap
    # on the number of candidates, and then do a greedy/local search assignment.

    rng = random.Random(42)

    # Enumerate candidates with max entry in [1, greedy_max]
    # For each max M, try to find random valid rows with that max.
    # We'll collect up to max_candidates rows.

    max_candidates = 5000
    candidates = []  # list of (max_entry, row, diff_set)

    # First, add the greedy rows as candidates
    for row in initial_rows:
        diffs = _get_row_diffs(row)
        candidates.append((row[-1], row, diffs))

    # Now enumerate random valid rows with max entry <= greedy_max
    # A valid row has all internal differences distinct (Sidon property within the row)
    # and starts at 0, strictly increasing.

    def _random_valid_row(max_entry, k, rng):
        """
        Generate a random valid row of length k+1 starting at 0, strictly increasing,
        with max entry <= max_entry, such that all internal differences are distinct.
        Returns None if not found (should be rare for reasonable max_entry).
        """
        if max_entry < k:  # Need at least k to fit k+1 distinct values from 0
            return None

        for _attempt in range(100):
            row = [0]
            used_diffs = set()
            success = True
            for i in range(1, k + 1):
                # Pick a random value in [row[-1]+1, max_entry]
                lo = row[-1] + 1
                hi = max_entry
                if lo > hi:
                    success = False
                    break
                x = rng.randint(lo, hi)
                new_diffs = {x - a for a in row}
                # Check distinctness within new_diffs (always true since a_i distinct)
                # Check disjoint from used_diffs
                if new_diffs & used_diffs:
                    success = False
                    break
                row.append(x)
                used_diffs |= new_diffs
            if success and len(row) == k + 1:
                return row
        return None

    # Enumerate candidates
    # Try max entries from small to greedy_max
    for M in range(k, greedy_max + 1):
        if time.time() > deadline - 10.0:
            break
        if len(candidates) >= max_candidates:
            break
        # Try to find a few random valid rows with max entry = M
        for _ in range(5):  # Up to 5 per M
            if time.time() > deadline - 10.0:
                break
            if len(candidates) >= max_candidates:
                break
            row = _random_valid_row(M, k, rng)
            if row is not None:
                diffs = _get_row_diffs(row)
                # Check if this diff set is already in candidates (avoid duplicates)
                # For efficiency, just add; duplicates are okay, assignment will handle
                candidates.append((row[-1], row, diffs))

    # Step 3: Solve assignment problem
    # We want to pick n_rows candidates with pairwise disjoint diff sets,
    # minimizing the maximum entry.

    # Approach: Sort candidates by max entry. For each possible scope S,
    # check if we can find n_rows candidates with max_entry <= S and pairwise
    # disjoint diff sets. Use a greedy + backtracking or matching approach.

    # Since n_rows is small (7), we can use a recursive backtracking search
    # over candidates sorted by max entry.

    # Sort candidates by max entry
    candidates.sort(key=lambda x: x[0])

    # Remove duplicates (same row)
    seen_rows = set()
    unique_candidates = []
    for max_e, row, diffs in candidates:
        key = tuple(row)
        if key not in seen_rows:
            seen_rows.add(key)
            unique_candidates.append((max_e, row, diffs))

    candidates = unique_candidates

    # Now try to find the minimum scope
    # Binary search on scope? Or just iterate?
    # Let's iterate from small to large.

    # For a given scope S, filter candidates with max_entry <= S
    # Then check if we can pick n_rows with disjoint diff sets.

    def _can_assign(cands, n_rows, k, rng, time_limit):
        """
        Given a list of candidates (max_e, row, diffs), check if we can pick
        n_rows with pairwise disjoint diff sets.
        Returns the assignment (list of rows) or None.
        Uses randomized greedy with backtracking.
        """
        start_t = time.time()
        # Sort by max_entry for diversity
        cands_sorted = sorted(cands, key=lambda x: x[0])

        # Greedy: pick n_rows with disjoint diffs
        # Use a randomized greedy: at each step, pick a random candidate
        # from those compatible with already picked.

        def _greedy_pick(cands, n_rows, rng):
            picked = []
            picked_diffs = set()
            available = list(cands)
            for _ in range(n_rows):
                if not available:
                    return None
                # Filter compatible
                compatible = []
                for c in available:
                    if c[2] & picked_diffs:
                        continue
                    compatible.append(c)
                if not compatible:
                    return None
                # Pick randomly from compatible, biased towards smaller max
                # Weight by 1/max_e to prefer smaller
                weights = [1.0 / max(1, c[0]) for c in compatible]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0.0
                chosen = compatible[-1]
                for i, c in enumerate(compatible):
                    cum += weights[i]
                    if r <= cum:
                        chosen = c
                        break
                picked.append(chosen)
                picked_diffs |= chosen[2]
            return [c[1] for c in picked]

        # Try multiple random greedy attempts
        best_assign = None
        best_scope = float('inf')
        for _attempt in range(200):
            if time.time() > time_limit:
                break
            # Shuffle candidates to get diversity
            shuffled = cands_sorted[:]
            rng.shuffle(shuffled)
            # But keep smaller max entries more likely to be picked
            # Actually, let's just try the greedy on the shuffled list
            assign = _greedy_pick(shuffled, n_rows, rng)
            if assign is not None:
                scope = max(row[-1] for row in assign)
                if scope < best_scope:
                    best_scope = scope
                    best_assign = assign

        return best_assign

    # Try to find the minimum scope
    # First, get an initial upper bound from greedy
    current_best_scope = best_score

    # Try scopes from small to current_best_scope
    # But we need to be efficient. Let's try a few key scopes.

    # Get all unique max entries from candidates
    unique_maxes = sorted(set(c[0] for c in candidates))

    # Binary search on scope
    lo = 0
    hi = len(unique_maxes) - 1
    best_assign = None
    best_scope = current_best_scope

    # First, try the current best scope to see if we can do better
    # Actually, let's just iterate from small to large and stop when we find a valid assignment
    # with scope < current_best_scope

    for S in unique_maxes:
        if S >= current_best_scope:
            break
        if time.time() > deadline - 5.0:
            break
        # Filter candidates with max_entry <= S
        filtered = [c for c in candidates if c[0] <= S]
        if len(filtered) < n_rows:
            continue
        # Try to assign
        assign = _can_assign(filtered, n_rows, k, rng, deadline - 2.0)
        if assign is not None:
            current_best_scope = S
            current_best_assign = assign
            # Continue to see if we can do even better
            # But since we're iterating in increasing order, this is the best for this S
            # We can break if we want, but let's continue to find potentially better
            # Actually, since we're going in increasing order, the first valid S is the minimum.
            # But our _can_assign might not find the optimal assignment for a given S.
            # So let's just record and continue.
            best_assign = assign
            best_scope = S

    # If we found a better assignment, update
    if best_assign is not None and best_scope < best_score:
        best_score = best_scope
        best_rows = [row[:] for row in best_assign]

    # Step 4: Local search to improve further
    # Use the current best as starting point, do row-level local search
    # to see if we can improve.

    current_rows = [row[:] for row in best_rows]
    row_diffs = [_get_row_diffs(r) for r in current_rows]
    current_score = _get_scope(current_rows)

    # Local search: try to replace each row with a candidate that has
    # smaller max entry and is compatible with other rows.

    T = 5.0
    cooling_rate = 0.99
    T_min = 0.1
    no_improve_count = 0
    max_no_improve = 30

    while time.time() < deadline:
        # Pick a random row to modify
        idx = random.randint(0, n_rows - 1)
        current_row = current_rows[idx]
        current_max = current_row[-1]

        # Find candidates compatible with all other rows
        other_diffs = set()
        for j in range(n_rows):
            if j != idx:
                other_diffs |= row_diffs[j]

        # Find compatible candidates with max < current_max
        compatible = []
        for max_e, row, diffs in candidates:
            if max_e >= current_max:
                continue
            if diffs & other_diffs:
                continue
            compatible.append((max_e, row, diffs))

        if not compatible:
            # No improvement possible for this row, try a random perturbation
            # Just pick a random compatible candidate (might have same or higher max)
            compatible_any = []
            for max_e, row, diffs in candidates:
                if diffs & other_diffs:
                    continue
                compatible_any.append((max_e, row, diffs))
            if not compatible_any:
                continue
            # Pick randomly
            chosen = random.choice(compatible_any)
            new_row = chosen[1]
            new_diffs = chosen[2]
            new_max = chosen[0]

            other_maxes = [current_rows[j][-1] for j in range(n_rows) if j != idx]
            other_max = max(other_maxes) if other_maxes else 0
            new_global_scope = max(new_max, other_max)
            delta = new_global_scope - current_score

            accept = False
            if delta < 0:
                accept = True
            elif T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True

            if accept:
                current_rows[idx] = new_row
                row_diffs[idx] = new_diffs
                current_score = new_global_scope
                if current_score < best_score:
                    best_score = current_score
                    best_rows = [row[:] for row in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                T = max(T_min, T * cooling_rate)
            else:
                no_improve_count += 1
                T = max(T_min, T * cooling_rate)
            continue

        # Pick the best compatible candidate (smallest max)
        # Or pick randomly from top few for diversity
        compatible.sort(key=lambda x: x[0])
        top_n = min(5, len(compatible))
        chosen = random.choice(compatible[:top_n])
        new_row = chosen[1]
        new_diffs = chosen[2]
        new_max = chosen[0]

        other_maxes = [current_rows[j][-1] for j in range(n_rows) if j != idx]
        other_max = max(other_maxes) if other_maxes else 0
        new_global_scope = max(new_max, other_max)
        delta = new_global_scope - current_score

        accept = False
        if delta < 0:
            accept = True
        elif T > T_min:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                accept = True

        if accept:
            current_rows[idx] = new_row
            row_diffs[idx] = new_diffs
            current_score = new_global_scope
            if current_score < best_score:
                best_score = current_score
                best_rows = [row[:] for row in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
            T = max(T_min, T * cooling_rate)
        else:
            no_improve_count += 1
            T = max(T_min, T * cooling_rate)

        if no_improve_count > max_no_improve:
            T = 5.0
            no_improve_count = 0

    return best_rows

def _greedy_construct(n_rows, k):
    """
    Greedy construction of rows.
    """
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

def _get_scope(rows):
    """
    Returns the maximum entry in any row.
    """
    scope = 0
    for row in rows:
        if row:
            scope = max(scope, row[-1])
    return scope

def _get_row_diffs(row):
    """
    Compute the set of all positive differences in a row.
    """
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

# EVOLVE-BLOCK-END
