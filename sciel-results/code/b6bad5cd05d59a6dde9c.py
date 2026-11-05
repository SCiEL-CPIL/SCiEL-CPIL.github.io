import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    # Phase 1: Generate initial solutions with backtracking + greedy hybrid
    seed = 0
    while time.time() < deadline - 80:  # Reserve 80s for SA + polish
        # Use backtracking for first few rows, then greedy
        rows = _hybrid_construct(n_rows, k, seed)
        scope = _scope(rows)

        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope == lower_bound:
                break

        seed += 1

    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)

    # Phase 2: Simulated annealing to improve (fast O(1) diff check)
    if time.time() < deadline - 20:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Dedicated shrink-the-max phase
    if time.time() < deadline - 5 and best_rows is not None:
        shrunk = _shrink_max_phase(best_rows, n_rows, k, start_time, deadline)
        if shrunk is not None:
            shr_scope = _scope(shrunk)
            if shr_scope < best_scope:
                best_scope = shr_scope
                best_rows = shrunk

    # Phase 4: Final polish pass targeting the maximum element
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _hybrid_construct(n_rows, k, seed):
    """
    Hybrid: use backtracking with forward checking for the first few rows,
    then switch to greedy packing for the rest.
    """
    rng = random.Random(seed)
    used = set()
    rows = []

    # Use backtracking for first 3 rows (or all if n_rows <= 3)
    bt_rows = min(3, n_rows)

    for row_idx in range(n_rows):
        if row_idx < bt_rows:
            # Use backtracking with forward checking
            row = _backtrack_row(used, k, rng, time_limit=5.0)
        else:
            # Use greedy packing
            row = _greedy_row(used, k, rng)

        if row is None:
            # Fallback to pure greedy
            row = _greedy_row(used, k, rng)

        if row is None:
            return None

        rows.append(row)
        # Update used differences
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                used.add(row[j] - row[i])

    return rows

def _backtrack_row(used_diffs, k, rng, time_limit=5.0):
    """
    Backtracking search for a single row with forward checking.
    Tries to find a row of k+1 increasing integers starting at 0
    whose differences don't conflict with used_diffs.
    """
    start_time = time.time()
    row = [0]
    # We need k more elements
    # Use iterative deepening or just DFS with pruning

    def get_internal_diffs(r):
        diffs = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
        return diffs

    def backtrack(current_row, depth):
        nonlocal start_time, used_diffs, k, rng

        if time.time() - start_time > time_limit:
            return None

        if depth == k:
            return list(current_row)

        last_val = current_row[-1]
        
        # Calculate differences already present in current_row
        # to ensure internal distinctness
        current_internal_diffs = get_internal_diffs(current_row)
        
        # Try candidates in ascending order, but with some randomization
        # to escape local minima
        candidates = []
        x = last_val + 1
        # Limit search space for efficiency
        max_x = last_val + 200
        while x <= max_x:
            new_diffs = {x - a for a in current_row}
            # Check against global used diffs
            if new_diffs & used_diffs:
                x += 1
                continue
            # Check against internal diffs of current_row
            if new_diffs & current_internal_diffs:
                x += 1
                continue
            candidates.append(x)
            x += 1

        if not candidates:
            return None

        # Randomize order slightly to get different solutions
        if rng.random() < 0.3:
            # Shuffle a subset
            if len(candidates) > 5:
                idx = rng.randint(0, len(candidates) - 1)
                candidates[idx], candidates[0] = candidates[0], candidates[idx]

        for x in candidates:
            current_row.append(x)
            # No need to re-check new_diffs against used_diffs here as done above,
            # but we must recurse. The check was done before appending.
            
            result = backtrack(current_row, depth + 1)
            if result is not None:
                return result

            current_row.pop()

        return None

    return backtrack(row, 0)

def _greedy_row(used_diffs, k, rng):
    """Best-fit greedy for a single row."""
    row = [0]
    used_sorted = sorted(used_diffs)

    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 500

        x = last_x + 1
        while len(valid_xs) < 40 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1

        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 40:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1

        if not valid_xs:
            return None

        # Best-fit: sort by packing score, pick from top few
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(5, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)

    return row

def _greedy_construct(n_rows, k, seed):
    """Pure best-fit greedy for all rows."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []

    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        if row is None:
            return None
        rows.append(row)
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                used.add(d)
                bisect.insort(used_sorted, d)

    return rows

def _packing_score(new_diffs, used, x):
    """
    Score a candidate x by how well its differences 'pack' into the used difference space.
    Lower score = better fit (less fragmentation).
    """
    if not used:
        return x * 1.0

    if not isinstance(used, list):
        used_sorted = sorted(used)
    else:
        used_sorted = used

    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist

    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search with maintained diff-count for O(1) validity checks,
    plus a dedicated shrink-the-max-row phase.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute difference sets for each row
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    # Maintain a global count of how many rows use each difference value.
    # A difference is "available" for a new row iff its count is 0.
    diff_count = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _row_is_valid(new_diffs, exclude_idx):
        """Check new_diffs doesn't conflict with any other row's diffs."""
        for d in new_diffs:
            # diff_count[d] includes the excluded row's contribution if present
            cnt = diff_count.get(d, 0)
            # If the excluded row uses d, it contributes 1 to cnt
            if d in row_diffs[exclude_idx]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit_row(row_idx, new_row, new_diffs):
        """Update rows, row_diffs, diff_count after accepting a new row."""
        nonlocal rows, row_diffs, diff_count
        # Remove old row's diffs from count
        for d in row_diffs[row_idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        # Add new row's diffs
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[row_idx] = new_row
        row_diffs[row_idx] = new_diffs
    
    # Phase A: General local search — try to improve any row
    T_start = 20.0
    T_end = 0.3
    total_steps = 40000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    T = T_start
    steps_since_improve = 0
    max_no_improve = 4000
    
    for step in range(total_steps):
        if time.time() > deadline - 20:
            break
        
        # 40% of the time, target the row with the largest last element
        if rng.random() < 0.4:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        new_row = _generate_candidate_row_fast(rows, row_idx, k, rng, row_diffs, diff_count)
        
        if new_row is None:
            continue
        
        n = len(new_row)
        new_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                new_diffs.add(new_row[j] - new_row[i])
        
        # O(|new_diffs|) conflict check using diff_count
        if not _row_is_valid(new_diffs, row_idx):
            continue
        
        # Accept: update diff_count by removing old row's diffs and adding new
        _commit_row(row_idx, new_row, new_diffs)
        
        new_scope = _scope(rows)
        delta = new_scope - current_scope
        if delta <= 0:
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_scope = new_scope
                steps_since_improve += 1
            else:
                # Reject: revert diff_count
                for d in new_diffs:
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        del diff_count[d]
                for d in row_diffs[row_idx]:
                    diff_count[d] = diff_count.get(d, 0) + 1
                rows[row_idx] = rows[row_idx]  # no change needed, but keep structure
                steps_since_improve += 1
        
        T *= cool_rate
        
        if steps_since_improve > max_no_improve:
            # Perturbation: pick a random row and rebuild it
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row_fast(rows, pert_idx, k, rng, row_diffs, diff_count)
            if pert_row is not None:
                n = len(pert_row)
                pert_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        pert_diffs.add(pert_row[j] - pert_row[i])
                
                if _row_is_valid(pert_diffs, pert_idx):
                    _commit_row(pert_idx, pert_row, pert_diffs)
                    current_scope = _scope(rows)
            steps_since_improve = 0
    
    # Phase B: Dedicated shrink-the-max phase
    if time.time() < deadline - 5:
        _shrink_max_internal(rows, row_diffs, diff_count, n_rows, k, rng, deadline)
        new_scope = _scope(rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in rows]
    
    return best_rows if best_scope < _scope(initial_rows) else None

def _generate_candidate_row_fast(rows, exclude_idx, k, rng, row_diffs, diff_count):
    """
    Generate a new candidate row using best-fit packing heuristic,
    with O(1) conflict checks using diff_count.
    """
    # Collect used diffs from other rows (excluding exclude_idx)
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 500
        
        x = last_x + 1
        while len(valid_xs) < 30 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 30:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)
    
    return row

def _shrink_max_internal(rows, row_diffs, diff_count, n_rows, k, rng, deadline):
    """Dedicated phase: iteratively try to reduce the maximum element by rebuilding the max row with a tighter bound."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline - 3:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to build a row with max element < current_max
        # Use progressively tighter target
        target = current_max - 1
        found = False
        for t in range(8):
            if time.time() > deadline - 3:
                break
            row = _build_row_constrained_fast(rows, max_idx, k, rng, target, row_diffs, diff_count)
            if row is not None and row[-1] < current_max:
                # Check conflicts with O(1) diff_count
                n = len(row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(row[j] - row[i])
                
                conflict = False
                for d in new_diffs:
                    cnt = diff_count.get(d, 0)
                    if d in row_diffs[max_idx]:
                        cnt -= 1
                    if cnt > 0:
                        conflict = True
                        break
                
                if not conflict:
                    # Accept
                    for d in row_diffs[max_idx]:
                        diff_count[d] -= 1
                        if diff_count[d] == 0:
                            del diff_count[d]
                    for d in new_diffs:
                        diff_count[d] = diff_count.get(d, 0) + 1
                    
                    rows[max_idx] = row
                    row_diffs[max_idx] = new_diffs
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    found = True
                    break
            
            # Relax target slightly
            target = min(target + 2, current_max - 1)
        
        if not found:
            # No improvement this round; try a few more with randomization
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                row = _build_row_constrained_fast(rows, max_idx, k, rng, current_max - 1, row_diffs, diff_count)
                if row is not None and row[-1] < current_max:
                    n = len(row)
                    new_diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            new_diffs.add(row[j] - row[i])
                    conflict = False
                    for d in new_diffs:
                        cnt = diff_count.get(d, 0)
                        if d in row_diffs[max_idx]:
                            cnt -= 1
                        if cnt > 0:
                            conflict = True
                            break
                    if not conflict:
                        for d in row_diffs[max_idx]:
                            diff_count[d] -= 1
                            if diff_count[d] == 0:
                                del diff_count[d]
                        for d in new_diffs:
                            diff_count[d] = diff_count.get(d, 0) + 1
                        rows[max_idx] = row
                        row_diffs[max_idx] = new_diffs
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                        found = True
                        break

def _build_row_constrained_fast(rows, exclude_idx, k, rng, target_max, row_diffs, diff_count):
    """Build a row of k+1 increasing ints starting at 0, all <= target_max,
    whose diffs don't conflict with other rows' diffs."""
    # Collect used diffs from other rows
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        
        x = last_x + 1
        while x <= target_max:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)
    
    if row[-1] <= target_max:
        return row
    return None

def _shrink_max_phase(rows, n_rows, k, start_time, deadline):
    """Dedicated phase: iteratively try to reduce the maximum element by rebuilding the max row with a tighter bound."""
    rng = random.Random(999)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)

    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)

    # Build diff_count for O(1) checks
    diff_count = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1

    _shrink_max_internal(rows, row_diffs, diff_count, n_rows, k, rng, deadline)

    orig_scope = _scope([list(r) for r in rows])
    if best_scope < orig_scope:
        return best_rows
    return None

def _polish_max(rows, n_rows, k, start_time, deadline):
    """Final polish: repeatedly try to reduce the maximum element."""
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    max_iter = 200

    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)

    diff_count = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1

    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break

        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        improved_row = _reduce_max_row(rows, max_idx, k, rng, current_max)

        if improved_row is not None and improved_row[-1] < current_max:
            n = len(improved_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(improved_row[j] - improved_row[i])

            conflict = False
            for d in new_diffs:
                cnt = diff_count.get(d, 0)
                if d in row_diffs[max_idx]:
                    cnt -= 1
                if cnt > 0:
                    conflict = True
                    break

            if not conflict:
                for d in row_diffs[max_idx]:
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        del diff_count[d]
                for d in new_diffs:
                    diff_count[d] = diff_count.get(d, 0) + 1

                new_rows = rows[:]
                new_rows[max_idx] = improved_row
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = new_rows
                    row_diffs[max_idx] = new_diffs

    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    """Try to find a replacement row with a smaller maximum element."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])

    used_sorted = sorted(used_diffs)

    target_max = current_max - 1
    for attempt in range(15):
        row = [0]
        success = True
        local_used = set(used_diffs)
        local_sorted = list(used_sorted)

        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]

            x = last_x + 1
            while x <= target_max:
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, local_sorted, x)
                    scored_xs.append((score, x))
                x += 1

            if not valid_xs:
                success = False
                break

            scored_xs.sort(key=lambda t: t[0])
            top_n = min(3, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            local_used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(local_sorted, d)

        if success and len(row) == k + 1:
            return row

        target_max = min(target_max + 2, current_max - 1)

    return None
