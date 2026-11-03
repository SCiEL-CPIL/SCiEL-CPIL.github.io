import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

def _build_diff_counter(rows):
    c = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            c[d] += 1
    return c

def _collision_count_from_counter(counter):
    return sum(max(0, v - 1) for v in counter.values())

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    """
    Fast greedy construction. For each row, greedily pick the smallest
    value that doesn't create a duplicate difference.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _backtrack_row(pos, current_row, current_diffs, used_diffs, max_val, deadline, rng, k):
    """
    Backtrack to build a single row of length k+1 starting at 0,
    with all diffs distinct and not in used_diffs, max value <= max_val.
    """
    n = k + 1
    if time.time() > deadline:
        return None
    if pos == n:
        return list(current_row)

    prev = current_row[-1]
    min_next = prev + 1
    # We need (n - pos - 1) more elements, each at least +1
    max_next = min(max_val, max_val - (n - pos - 1))
    if min_next > max_next:
        return None

    # Generate candidates: smaller values first (to minimize scope)
    # But add some randomness to avoid getting stuck
    candidates = list(range(min_next, max_next + 1))
    if len(candidates) > 10:
        # Shuffle the upper half for diversity
        mid = len(candidates) // 2
        rng.shuffle(candidates[mid:])

    for val in candidates:
        if time.time() > deadline:
            return None

        # Check conflicts
        conflict = False
        new_diffs = []
        for idx in range(pos):
            d = val - current_row[idx]
            if d in current_diffs or d in used_diffs:
                conflict = True
                break
            new_diffs.append(d)

        if conflict:
            continue

        current_row.append(val)
        current_diffs.update(new_diffs)

        result = _backtrack_row(pos + 1, current_row, current_diffs, used_diffs, max_val, deadline, rng, k)

        current_row.pop()
        for d in new_diffs:
            current_diffs.discard(d)

        if result is not None:
            return result

    return None

def _construct_rows_backtrack(n_rows, k, max_scope, deadline, rng):
    """
    Construct n_rows using backtracking, trying to keep max value <= max_scope.
    """
    used_diffs = set()
    rows = []

    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None

        # Try multiple attempts with increasing max_val
        best_row = None
        for attempt in range(30):
            if time.time() > deadline:
                return None

            current_row = [0]
            current_diffs = set()

            # Try max_val from max_scope down to a minimum
            # Actually, let's just try with max_scope first
            result = _backtrack_row(1, current_row, current_diffs, used_diffs, max_scope, deadline, rng, k)

            if result is not None:
                if best_row is None or result[-1] < best_row[-1]:
                    best_row = result
                # If we found a row with max <= max_scope, we're good
                # But we want the smallest possible, so keep searching a bit
                if result[-1] <= max_scope and attempt >= 5:
                    break

        if best_row is None:
            return None

        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)

    return rows

def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    """
    Simulated Annealing to minimize scope with collision penalty.
    Uses efficient incremental diff counting.
    """
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 5000000

    current = [list(r) for r in rows]
    current_scope = _scope(current)

    # Build diff counter
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count_from_counter(diff_counter)

    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions

    collision_weight = 100000

    def cost(scope, coll):
        return coll * collision_weight + scope

    T = 100.0
    cooling = 0.99995
    min_T = 0.01

    it = 0
    start_time = time.time()
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1

        # Choose random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)

        old_val = current[i][j]
        prev_val = current[i][j - 1]
        next_val = current[i][j + 1] if j < k else old_val + 10000

        # Move type
        move_type = rng.random()

        if move_type < 0.4:
            # Try to reduce (especially if near max)
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.7:
            # Small perturbation
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Larger move
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 20))

        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue

        # Compute old diffs for this row
        old_row = current[i]
        old_diffs = _diffs_for_row(old_row)

        # Apply change
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)

        # Check internal validity (all diffs positive and distinct within row)
        if len(new_diffs) != len(set(new_diffs)):
            current[i][j] = old_val
            continue

        # Update diff counter
        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]

        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1

        new_collisions = _collision_count_from_counter(diff_counter)

        # New scope
        new_scope = current_scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope and new_val < old_val:
            # Need to recompute
            new_scope = _scope(current)

        old_c = cost(current_scope, collisions)
        new_c = cost(new_scope, new_collisions)

        if new_c <= old_c:
            accept = True
        else:
            delta_c = new_c - old_c
            prob = math.exp(-delta_c / max(T, min_T))
            accept = rng.random() < prob

        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1

        T = max(T * cooling, min_T)

    return best_scope, best

def _scope_squeeze(rows, k, deadline, rng):
    """
    Try to reduce the scope by 1 iteratively using local search.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)

    # Precompute other rows' diff sets for faster checking
    other_diff_sets = []
    for i, r in enumerate(rows):
        s = _row_diffs_set(r)
        other_diff_sets.append(s)

    for _ in range(500):
        if time.time() > deadline:
            break

        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break

        i = rng.choice(max_rows)
        row = rows[i]
        old_max = row[-1]
        prev = row[k - 1]

        found = False
        for new_val in range(old_max - 1, prev, -1):
            row[k] = new_val
            new_diffs = _diffs_for_row(row)

            if len(new_diffs) != len(set(new_diffs)):
                continue

            # Check cross-row conflicts
            conflict = False
            new_diffs_set = set(new_diffs)
            for other_idx, other_set in enumerate(other_diff_sets):
                if other_idx == i:
                    continue
                for d in new_diffs_set:
                    if d in other_set:
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict:
                current_scope = new_val
                found = True
                # Update the diff set for this row
                other_diff_sets[i] = new_diffs_set
                break

        if not found:
            row[k] = old_max
            continue

    return rows

def _min_conflicts_search(n_rows, k, deadline, rng, initial_rows=None):
    """
    Min-Conflicts style local search: start from an initial assignment,
    then iteratively move variables to minimize conflicts.
    """
    if initial_rows is None:
        initial_rows = _fast_greedy(n_rows, k, rng)
        if initial_rows is None:
            return None

    current = [list(r) for r in initial_rows]
    diff_counter = _build_diff_counter(current)
    current_scope = _scope(current)

    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = _collision_count_from_counter(diff_counter)

    collision_weight = 50000

    def cost(scope, coll):
        return coll * collision_weight + scope

    current_collisions = _collision_count_from_counter(diff_counter)
    T = 50.0
    cooling = 0.9999
    min_T = 0.01

    max_iters = 3000000
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1

        # Choose a random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)

        old_val = current[i][j]
        prev_val = current[i][j - 1]
        next_val = current[i][j + 1] if j < k else old_val + 10000

        # Try several candidate values
        best_candidate = None
        best_candidate_cost = float('inf')

        # Generate candidates
        candidates = set()
        # Values near current
        for delta in range(-10, 11):
            candidates.add(old_val + delta)
        # Values that minimize conflicts: try small values
        for v in range(prev_val + 1, min(prev_val + 20, next_val)):
            candidates.add(v)
        # Random
        for _ in range(5):
            candidates.add(rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30)))

        for new_val in candidates:
            if new_val <= prev_val or (j < k and new_val >= next_val):
                continue
            if new_val == old_val:
                continue

            # Compute new diffs
            old_row = list(current[i])
            current[i][j] = new_val
            new_row = current[i]
            new_diffs = _diffs_for_row(new_row)

            if len(new_diffs) != len(set(new_diffs)):
                current[i][j] = old_val
                continue

            # Update counter
            old_diffs = _diffs_for_row(old_row)
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1

            new_collisions = _collision_count_from_counter(diff_counter)
            new_scope = _scope(current)
            c = cost(new_scope, new_collisions)

            if c < best_candidate_cost:
                best_candidate_cost = c
                best_candidate = new_val
                best_new_scope = new_scope
                best_new_collisions = new_collisions

            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1

        if best_candidate is None:
            continue

        # Accept with probability
        current[i][j] = best_candidate
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        old_row = list(current[i])
        old_row[j] = old_val
        old_diffs = _diffs_for_row(old_row)

        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1

        current_collisions = _collision_count_from_counter(diff_counter)
        current_scope = _scope(current)

        if current_scope < best_scope or (current_scope == best_scope and current_collisions < best_collisions):
            best_scope = current_scope
            best_collisions = current_collisions
            best = [list(r) for r in current]

        T = max(T * cooling, min_T)

    return best

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    rng = random.Random(12345)

    best_result = None
    best_score = float('inf')

    # Phase 1: Fast greedy to get a valid starting point
    rows = _fast_greedy(n_rows, k, rng)
    if rows is not None:
        best_result = rows
        best_score = _scope(rows)
    else:
        # Last resort: simple sequential
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows
        best_score = _scope(rows)

    # Phase 2: Backtracking construction with tighter scope
    if time.time() < deadline - 30:
        # Try to construct with scope = best_score - 1
        target_scope = best_score - 1
        rows_bt = _construct_rows_backtrack(n_rows, k, target_scope, deadline - 20, rng)
        if rows_bt is not None:
            s = _scope(rows_bt)
            if s < best_score:
                best_score = s
                best_result = rows_bt

    # Phase 3: Simulated Annealing optimization
    if time.time() < deadline - 15:
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.6
        max_iters = int(remaining * 30000)
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=max_iters)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows

    # Phase 4: Min-Conflicts search
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        mc_deadline = time.time() + remaining * 0.7
        rows_mc = _min_conflicts_search(n_rows, k, mc_deadline, rng, initial_rows=best_result)
        if rows_mc is not None and _valid(rows_mc, k):
            s = _scope(rows_mc)
            if s < best_score:
                best_score = s
                best_result = rows_mc

    # Phase 5: Scope squeeze
    if time.time() < deadline - 3:
        remaining = deadline - time.time()
        squeeze_deadline = time.time() + remaining * 0.8
        squeezed = _scope_squeeze(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed

    # Final validation
    if best_result is not None and not _valid(best_result, k):
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows

    return best_result
