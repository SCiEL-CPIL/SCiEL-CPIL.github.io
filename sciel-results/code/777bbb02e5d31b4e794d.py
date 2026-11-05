import time
import random
import math

# EVOLVE-BLOCK-START


def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used


def _greedy_construct(n_rows, k, seed_offset=0):
    """Greedy construction as a fast baseline."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows


def _count_future_diffs(row, used, max_val):
    """
    Count how many valid differences remain available for this row
    if we were to continue building it up to max_val.
    Lower count = more constrained.
    """
    count = 0
    last = row[-1]
    for val in range(last + 1, max_val + 1):
        new_diffs = {val - a for a in row}
        if not (new_diffs & used):
            count += 1
    return count


def _csp_interleaved(n_rows, k, max_scope, rng, deadline):
    """
    Constraint satisfaction approach: build all rows simultaneously.
    At each step, pick the most constrained incomplete row (fewest valid
    next candidates) and extend it. This is a structural change from
    building rows sequentially.
    
    The key insight: by always working on the most constrained row, we
    minimize the chance of getting stuck later when a row has no valid
    extensions.
    """
    if time.time() > deadline:
        return None
    
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        # Find the most constrained incomplete row
        min_count = float('inf')
        min_row = -1
        min_candidates = []
        
        for i in range(n_rows):
            if len(rows[i]) >= k + 1:
                continue
            low = rows[i][-1] + 1
            high = max_scope
            candidates = []
            for val in range(low, high + 1):
                new_diffs = {val - a for a in rows[i]}
                if not (new_diffs & used):
                    candidates.append((val, new_diffs))
            if len(candidates) < min_count:
                min_count = len(candidates)
                min_row = i
                min_candidates = candidates
        
        if min_row == -1:
            return True  # All rows complete
        
        if min_count == 0:
            return False  # Dead end
        
        # Try candidates for the most constrained row
        # Use a heuristic: prefer candidates that leave more options for future steps
        def candidate_score(val, new_diffs):
            # Score based on how many differences this consumes (fewer is better)
            # and how close to the end of the range it is (closer is better for scope)
            return (len(new_diffs), -val)
        
        min_candidates.sort(key=candidate_score)
        
        # Randomize occasionally to escape local optima
        if rng.random() < 0.2:
            rng.shuffle(min_candidates)
        
        for val, new_diffs in min_candidates:
            rows[min_row].append(val)
            used.update(new_diffs)
            if backtrack():
                return True
            used.difference_update(new_diffs)
            rows[min_row].pop()
        
        return False
    
    if backtrack():
        if all(len(rows[i]) == k + 1 for i in range(n_rows)):
            return [r[:] for r in rows]
    return None


def _csp_with_lookahead(n_rows, k, max_scope, rng, deadline):
    """
    Enhanced CSP with lookahead: when choosing a candidate, estimate how
    it affects the constraint tightness of other incomplete rows.
    Prefer candidates that keep other rows' options open.
    """
    if time.time() > deadline:
        return None
    
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    def estimate_constraint_tightness(row, used_set, max_val):
        """Estimate how constrained a row is given current used set."""
        if len(row) >= k + 1:
            return 0
        last = row[-1]
        count = 0
        for val in range(last + 1, min(last + 20, max_val + 1)):  # Look ahead 20 steps
            new_diffs = {val - a for a in row}
            if not (new_diffs & used_set):
                count += 1
        return count
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        # Find the most constrained incomplete row
        min_count = float('inf')
        min_row = -1
        min_candidates = []
        
        for i in range(n_rows):
            if len(rows[i]) >= k + 1:
                continue
            low = rows[i][-1] + 1
            high = max_scope
            candidates = []
            for val in range(low, high + 1):
                new_diffs = {val - a for a in rows[i]}
                if not (new_diffs & used):
                    candidates.append((val, new_diffs))
            if len(candidates) < min_count:
                min_count = len(candidates)
                min_row = i
                min_candidates = candidates
        
        if min_row == -1:
            return True  # All rows complete
        
        if min_count == 0:
            return False  # Dead end
        
        # Score candidates based on impact on other rows
        def candidate_impact(val, new_diffs):
            # Estimate how this choice affects other incomplete rows
            impact = 0
            for j in range(n_rows):
                if j == min_row or len(rows[j]) >= k + 1:
                    continue
                # Check if this choice blocks any immediate options for row j
                last_j = rows[j][-1]
                for next_val in range(last_j + 1, min(last_j + 10, max_scope + 1)):
                    new_diffs_j = {next_val - a for a in rows[j]}
                    if not (new_diffs_j & used) and (new_diffs_j & new_diffs):
                        impact += 1
            return impact
        
        min_candidates.sort(key=lambda x: (candidate_impact(x[0], x[1]), len(x[1])))
        
        if rng.random() < 0.15:
            rng.shuffle(min_candidates)
        
        for val, new_diffs in min_candidates:
            rows[min_row].append(val)
            used.update(new_diffs)
            if backtrack():
                return True
            used.difference_update(new_diffs)
            rows[min_row].pop()
        
        return False
    
    if backtrack():
        if all(len(rows[i]) == k + 1 for i in range(n_rows)):
            return [r[:] for r in rows]
    return None


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])

        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]

            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break

    return rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    for _ in range(3):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        if time.time() > deadline:
                            break
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break

    return rows


def _difference_partition_search(n_rows, k, max_scope, rng, deadline):
    """
    Structural change: model as partitioning the difference space.
    
    Instead of building rows element by element, we think about which
    differences each row should "own". Each row of k+1 elements has exactly
    k*(k+1)/2 differences. We need to partition a subset of {1, ..., max_scope}
    into n_rows groups, each of size k*(k+1)/2, such that each group can be
    realized as the difference set of a valid row.
    
    This is a different modeling family: we're assigning differences to rows
    first, then checking if each assignment is realizable.
    """
    if time.time() > deadline:
        return None
    
    diffs_per_row = k * (k + 1) // 2
    total_needed = n_rows * diffs_per_row
    
    if max_scope < total_needed:
        return None
    
    # For each candidate difference d, determine which rows could potentially use it
    # based on feasibility constraints
    
    # Start with a greedy assignment: assign differences to rows in a balanced way
    # Then check if each row's assigned differences are realizable
    
    # A row with differences D is realizable if there exists a strictly increasing
    # sequence a_0=0 < a_1 < ... < a_k such that {a_j - a_i : i<j} = D
    
    # This is a complex constraint. Instead, use a heuristic:
    # 1. Start with a valid solution from greedy
    # 2. Try to improve by swapping differences between rows
    
    g = _greedy_construct(n_rows, k)
    if not g:
        return None
    
    rows = [r[:] for r in g]
    used = _all_diffs_used(rows)
    
    # Try to improve by swapping elements between rows
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_rows):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_rows):
                if time.time() > deadline:
                    break
                # Try swapping last elements of rows i and j
                row_i = rows[i]
                row_j = rows[j]
                
                old_last_i = row_i[-1]
                old_last_j = row_j[-1]
                
                # Try swapping
                row_i[-1] = old_last_j
                row_j[-1] = old_last_i
                
                # Check validity
                valid = True
                all_diffs = set()
                for r in rows:
                    if r[0] != 0:
                        valid = False
                        break
                    for p in range(len(r) - 1):
                        if r[p + 1] <= r[p]:
                            valid = False
                            break
                    if not valid:
                        break
                    rd = _row_diffs(r)
                    if rd & all_diffs:
                        valid = False
                        break
                    all_diffs |= rd
                
                if valid:
                    scope = _total_scope(rows)
                    if scope <= max_scope:
                        improved = True
                else:
                    # Undo
                    row_i[-1] = old_last_i
                    row_j[-1] = old_last_j
    
    if time.time() < deadline:
        # Local search to reduce scope
        rows = _local_search_scope(rows, k, rng, deadline)
    
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    
    Uses multiple structural approaches:
    1. CSP interleaved search (build most constrained row first)
    2. CSP with lookahead (consider impact on other rows)
    3. Difference partition search (model as partitioning difference space)
    4. Local search and polish for refinement
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Get a greedy baseline for initial scope estimate
    g = _greedy_construct(n_rows, k)
    g_scope = _total_scope(g)
    best_rows = [r[:] for r in g]
    best_scope = g_scope

    # Phase 1: CSP interleaved search with tightening bounds
    csp_deadline = deadline - 50
    for max_val in range(best_scope, 104, -1):
        if time.time() > csp_deadline:
            break
        for attempt in range(3):
            if time.time() > csp_deadline:
                break
            attempt_rng = random.Random(5000 * max_val + attempt + 13)
            result = _csp_interleaved(n_rows, k, max_val, attempt_rng, csp_deadline)
            if result is not None and len(result) == n_rows:
                scope = _total_scope(result)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in result]
                if max_val == scope:
                    break
        if best_scope == max_val:
            break

    # Phase 2: CSP with lookahead
    if time.time() < deadline - 40:
        la_deadline = deadline - 40
        for max_val in range(best_scope, 104, -1):
            if time.time() > la_deadline:
                break
            for attempt in range(2):
                if time.time() > la_deadline:
                    break
                attempt_rng = random.Random(7000 * max_val + attempt + 23)
                result = _csp_with_lookahead(n_rows, k, max_val, attempt_rng, la_deadline)
                if result is not None and len(result) == n_rows:
                    scope = _total_scope(result)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [r[:] for r in result]
                    if max_val == scope:
                        break
            if best_scope == max_val:
                break

    # Phase 3: Difference partition search
    if time.time() < deadline - 25:
        dp_deadline = deadline - 25
        for max_val in range(best_scope, 104, -1):
            if time.time() > dp_deadline:
                break
            for attempt in range(2):
                if time.time() > dp_deadline:
                    break
                attempt_rng = random.Random(9000 * max_val + attempt + 33)
                result = _difference_partition_search(n_rows, k, max_val, attempt_rng, dp_deadline)
                if result is not None and len(result) == n_rows:
                    scope = _total_scope(result)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [r[:] for r in result]
                    if max_val == scope:
                        break
            if best_scope == max_val:
                break

    # Phase 4: Local search to reduce scope
    if time.time() < deadline - 15:
        ls_rows = _local_search_scope(best_rows, k, rng, deadline - 15)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Phase 5: Final polish
    if time.time() < deadline - 3:
        polish_rows = _polish_pass(best_rows, k, deadline - 3)
        polish_scope = _total_scope(polish_rows)
        if polish_scope < best_scope:
            best_rows = polish_rows

    return best_rows

# EVOLVE-BLOCK-END
