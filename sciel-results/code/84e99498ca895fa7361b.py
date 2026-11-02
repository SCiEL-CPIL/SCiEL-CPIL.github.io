import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
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

def _greedy_init_random(n_rows, k, used_set, max_val, rng):
    """
    Greedy init with random tie-breaking among valid candidates.
    used_set: set of already-used differences.
    max_val: upper bound for entries.
    Returns list of rows or None if stuck.
    """
    rows = []
    for r in range(n_rows):
        row = [0]
        x = 0
        stuck = False
        while len(row) < k + 1:
            found = False
            # Try a window of candidates starting from x+1
            for attempt in range(max_val - x):
                cand = x + 1 + attempt
                new_diffs = {cand - a for a in row}
                if not new_diffs & used_set:
                    row.append(cand)
                    used_set |= new_diffs
                    x = cand
                    found = True
                    break
            if not found:
                stuck = True
                break
        if stuck:
            return None
        rows.append(row)
    return rows

def _backtrack_row(k, used_set, max_val, rng, n_restarts=5):
    """
    Backtracking search for a single valid row of k+1 entries starting at 0.
    Returns the row or None.
    """
    for _restart in range(n_restarts):
        row = [0]
        used_local = set()
        success = _backtrack_row_dfs(row, k, used_set, used_local, max_val, rng)
        if success:
            return row
    return None

def _backtrack_row_dfs(row, k, used_set, used_local, max_val, rng, depth=0):
    """DFS backtracking for one row."""
    if len(row) == k + 1:
        return True
    if depth > 50000:
        return False
    x = row[-1]
    # Generate candidates in random order for diversity
    cands = list(range(x + 1, min(x + 80, max_val + 1)))
    rng.shuffle(cands)
    for cand in cands:
        new_diffs = {cand - a for a in row}
        if new_diffs & (used_set | used_local):
            continue
        row.append(cand)
        used_local |= new_diffs
        if _backtrack_row_dfs(row, k, used_set, used_local, max_val, rng, depth + 1):
            return True
        row.pop()
        used_local -= new_diffs
    return False

def _row_diffs_set(row):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
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

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used_set(rows):
    """Compute the union of all differences across all rows (as a set)."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _compute_all_used_ba(rows, max_diff):
    """Compute the union of all differences across all rows as a bytearray."""
    ba = bytearray(max_diff + 1)
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d <= max_diff:
                    ba[d] = 1
    return ba

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _build_static_diffs(row, k, pair_idx):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for (i, j) in pair_idx:
            if i == pi or j == pi:
                continue
            s.add(row[j] - row[i])
        static[pi] = s
    return static

def _pair_indices(k):
    """Precompute (i, j) pairs for a row of length k+1."""
    n = k + 1
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _try_move_fast(current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                   row_maxes, k, max_diff):
    """
    Try to apply a move. Returns (success, new_diffs, cand_row_max, new_global_max).
    """
    row = current_rows[ri]
    if new_val <= row[pi - 1]:
        return False, None, None, None
    if pi < k and new_val >= row[pi + 1]:
        return False, None, None, None

    static_pi = static_diffs[ri][pi]
    old_diffs = row_diffs[ri]

    # Compute the k diffs involving pi and check for internal conflicts
    new_diffs = set(static_pi)
    valid = True
    for i in range(len(row)):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)

    if not valid:
        return False, None, None, None

    # Check for conflicts with other rows via the bytearray
    for d in new_diffs:
        if d in old_diffs:
            continue
        if d <= max_diff and all_used_ba[d]:
            return False, None, None, None

    if pi == k:
        cand_row_max = new_val
    else:
        cand_row_max = row[k]

    old_row_max = row_maxes[ri]
    row_maxes[ri] = cand_row_max
    new_global_max = max(row_maxes)
    row_maxes[ri] = old_row_max

    return True, new_diffs, cand_row_max, new_global_max

def _apply_move_fast(current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                     row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                     pair_idx, max_diff):
    """Apply a valid move to the state."""
    for d in old_diffs:
        if d <= max_diff:
            all_used_ba[d] = 0
    for d in new_diffs:
        if d <= max_diff:
            all_used_ba[d] = 1
    current_rows[ri][pi] = new_val
    row_diffs[ri] = new_diffs
    row_maxes[ri] = cand_row_max
    static_diffs[ri] = _build_static_diffs(current_rows[ri], k, pair_idx)
    return new_global_max if False else max(row_maxes)

def _sa_phase(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
              current_val, best_rows, best_val, T, T_min, cool, max_no_improve,
              deadline, pair_idx, max_diff):
    """Run simulated annealing phase. Returns updated state."""
    no_improve = 0

    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]

        # Generate candidates
        candidates = []

        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)

        # Increasing candidates
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
        max_increase = min(20, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)

        # Random jump
        if random.random() < 0.2:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            if low <= high:
                candidates.append(random.randint(low, high))

        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        random.shuffle(candidates)

        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                row_maxes, k, max_diff)

            if not success:
                continue

            delta = new_global_max - current_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.25:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True

            if accept:
                current_val = _apply_move_fast(
                    current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                    row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                    pair_idx, max_diff)

                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1

                moved = True
                break

        if not moved:
            no_improve += 1

        T = max(T_min, T * cool)

        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used_ba = _compute_all_used_ba(current_rows, max_diff)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0

    return current_rows, current_val, all_used_ba, row_diffs, static_diffs, row_maxes, best_rows, best_val

def _polish_phase(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
                  current_val, best_rows, best_val, deadline, pair_idx, max_diff):
    """Final polishing with only decreasing moves and tight tolerance."""
    T = 1.0
    T_min = 0.05
    cool = 0.99998
    no_improve = 0
    max_no_improve_polish = 15000

    while time.time() < deadline and no_improve < max_no_improve_polish:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]

        # Only decreasing moves in polish
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(25, old_val - min_allowed)

        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)

        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        random.shuffle(candidates)

        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                row_maxes, k, max_diff)

            if not success:
                continue

            delta = new_global_max - current_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.15:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.03))
                    if random.random() < prob:
                        accept = True

            if accept:
                current_val = _apply_move_fast(
                    current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                    row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                    pair_idx, max_diff)

                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1

                moved = True
                break

        if not moved:
            no_improve += 1

        T = max(T_min, T * cool)

    return best_rows, best_val

def _exact_descent_polish(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs,
                          row_maxes, current_val, best_rows, best_val, deadline,
                          pair_idx, max_diff):
    """
    Deterministic local solve: repeatedly scan every (row, position) and apply the
    first improving (or neutral) move found. Continue until a full pass makes no
    change or the deadline is reached. This is a tighter, exact local optimum
    search that complements the stochastic polish.
    """
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for ri in range(n_rows):
            if time.time() >= deadline:
                break
            for pi in range(1, k + 1):
                if time.time() >= deadline:
                    break
                row = current_rows[ri]
                old_val = row[pi]
                old_diffs = row_diffs[ri]

                min_allowed = row[pi - 1] + 1
                max_allowed = (row[pi + 1] - 1) if pi < k else max_diff
                # Try decreasing values first (most likely to improve the max)
                for new_val in range(min_allowed, old_val):
                    success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                        current_rows, ri, pi, new_val, static_diffs, row_diffs,
                        all_used_ba, row_maxes, k, max_diff)
                    if not success:
                        continue
                    if new_global_max < current_val:
                        current_val = _apply_move_fast(
                            current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                            row_diffs, all_used_ba, row_maxes, static_diffs, k,
                            old_diffs, pair_idx, max_diff)
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        progress = True
                        break
                # Also try a few increasing values (may free up space for other rows)
                for new_val in range(old_val + 1, min(old_val + 10, max_allowed + 1)):
                    success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                        current_rows, ri, pi, new_val, static_diffs, row_diffs,
                        all_used_ba, row_maxes, k, max_diff)
                    if not success:
                        continue
                    if new_global_max < current_val:
                        current_val = _apply_move_fast(
                            current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                            row_diffs, all_used_ba, row_maxes, static_diffs, k,
                            old_diffs, pair_idx, max_diff)
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        progress = True
                        break

    return best_rows, best_val

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used_set(rows):
    """Compute the union of all differences across all rows (as a set)."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _validate_rows(rows, k):
    """Validate that rows are proper difference sets."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def _row_reconstruction_sa(current_rows, k, n_rows, best_rows, best_val, deadline, rng):
    """
    SA that replaces entire rows using backtracking.
    Each step: pick a row to remove, rebuild it with backtracking given the
    remaining rows' differences. Accept if it improves or is neutral.
    """
    current_val = _global_max(current_rows)
    no_improve = 0
    T = 100.0
    T_min = 1.0
    cool = 0.9999

    while time.time() < deadline and no_improve < 20000:
        # Pick a row to reconstruct
        ri = rng.randrange(n_rows)
        old_row = current_rows[ri]
        old_row_max = max(old_row)

        # Compute differences used by all OTHER rows
        other_diffs = set()
        for j in range(n_rows):
            if j == ri:
                continue
            other_diffs |= _row_diffs_set(current_rows[j])

        # Try to rebuild row ri with backtracking, bounded by current scope
        scope_bound = max(current_val, best_val) + 20
        new_row = _backtrack_row(k, other_diffs, scope_bound, rng, n_restarts=3)

        if new_row is None:
            # Try with a slightly larger bound
            new_row = _backtrack_row(k, other_diffs, scope_bound + 30, rng, n_restarts=2)

        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        new_row_max = max(new_row)
        # Compute new global max
        other_maxes = [max(current_rows[j]) for j in range(n_rows) if j != ri]
        new_global = max([new_row_max] + other_maxes) if other_maxes else new_row_max

        delta = new_global - current_val

        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if rng.random() < 0.3:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta / max(T, 0.1))
                if rng.random() < prob:
                    accept = True

        if accept:
            current_rows[ri] = new_row
            current_val = new_global
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T = max(T_min, T * cool)

        # Periodic restart
        if no_improve > 5000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            T = 100.0
            no_improve = 0

    return current_rows, current_val, best_rows, best_val

def _pairwise_swap_sa(current_rows, k, n_rows, best_rows, best_val, deadline, rng):
    """
    SA with pairwise row swaps: swap two rows and check validity.
    Combined with row reconstruction for the swapped-out row.
    """
    current_val = _global_max(current_rows)
    no_improve = 0
    T = 50.0
    T_min = 0.5
    cool = 0.99995

    while time.time() < deadline and no_improve < 30000:
        # Pick two distinct rows
        ri = rng.randrange(n_rows)
        rj = rng.randrange(n_rows - 1)
        if rj >= ri:
            rj += 1

        old_row_i = current_rows[ri]
        old_row_j = current_rows[rj]

        # Try swapping: remove row j's diffs, add row i's, then try to rebuild row j
        # Actually, let's try a different strategy: rebuild one row given the other
        # Remove row i, rebuild it given all other rows (including j)
        other_diffs = set()
        for m in range(n_rows):
            if m == ri:
                continue
            other_diffs |= _row_diffs_set(current_rows[m])

        scope_bound = max(current_val, best_val) + 10
        new_row = _backtrack_row(k, other_diffs, scope_bound, rng, n_restarts=2)

        if new_row is not None:
            new_row_max = max(new_row)
            other_maxes = [max(current_rows[m]) for m in range(n_rows) if m != ri]
            new_global = max([new_row_max] + other_maxes) if other_maxes else new_row_max

            delta = new_global - current_val
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if rng.random() < prob:
                        accept = True

            if accept:
                current_rows[ri] = new_row
                current_val = new_global
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1

        T = max(T_min, T * cool)

        if no_improve > 8000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            T = 50.0
            no_improve = 0

    return current_rows, current_val, best_rows, best_val

def _local_descent(current_rows, k, n_rows, deadline, rng):
    """
    Deterministic local descent: for each row, try to find a smaller valid row
    via backtracking with a tighter bound.
    """
    best_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]

    progress = True
    while progress and time.time() < deadline:
        progress = False
        for ri in range(n_rows):
            if time.time() >= deadline:
                break
            old_row = current_rows[ri]
            old_max = max(old_row)

            other_diffs = set()
            for m in range(n_rows):
                if m == ri:
                    continue
                other_diffs |= _row_diffs_set(current_rows[m])

            # Try to find a row with smaller max
            target_max = old_max - 1
            new_row = _backtrack_row(k, other_diffs, target_max, rng, n_restarts=10)

            if new_row is not None:
                new_max = max(new_row)
                if new_max < old_max:
                    current_rows[ri] = new_row
                    new_global = _global_max(current_rows)
                    if new_global < best_val:
                        best_val = new_global
                        best_rows = [r[:] for r in current_rows]
                    progress = True

    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    rng = random.Random(42)

    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)

    # 2. Row reconstruction SA - Phase 1 (exploration)
    current_rows = [r[:] for r in rows]
    phase1_deadline = start_time + 50.0
    current_rows, current_val, best_rows, best_val = _row_reconstruction_sa(
        current_rows, k, n_rows, best_rows, best_val, phase1_deadline, rng)

    # 3. Row reconstruction SA - Phase 2 (exploitation)
    if time.time() < deadline - 30:
        current_rows = [r[:] for r in best_rows]
        phase2_deadline = start_time + 80.0
        current_rows, current_val, best_rows, best_val = _row_reconstruction_sa(
            current_rows, k, n_rows, best_rows, best_val, phase2_deadline, rng)

    # 4. Local descent polish
    if time.time() < deadline - 10:
        current_rows = [r[:] for r in best_rows]
        polish_deadline = deadline - 3.0
        best_rows, best_val = _local_descent(
            current_rows, k, n_rows, polish_deadline, rng)

    # Final validation
    if not _validate_rows(best_rows, k):
        # Fallback to greedy
        best_rows = _greedy_init(n_rows, k)

    return best_rows

# EVOLVE-BLOCK-END
