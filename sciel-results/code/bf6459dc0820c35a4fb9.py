"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound on max entry
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    # Compact representation: each row stored as list of k non-zero values
    # Full row = [0] + compact_row

    def to_full(compact_row):
        return [0] + list(compact_row)

    def compute_diffs(compact_row):
        """Compute all positive differences for a compact row [a1, a2, ..., ak]."""
        full = [0] + list(compact_row)
        diffs = set()
        for i in range(len(full)):
            for j in range(i + 1, len(full)):
                diffs.add(full[j] - full[i])
        return diffs

    def is_valid(compact_rows):
        """Check if all differences are distinct across all rows."""
        used = set()
        for cr in compact_rows:
            full = [0] + list(cr)
            for i in range(len(full)):
                for j in range(i + 1, len(full)):
                    d = full[j] - full[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_max_entry(compact_rows):
        return max(max(cr) for cr in compact_rows)

    def greedy_initial(n_rows, k, rng):
        """Generate an initial valid solution using greedy."""
        used = set()
        compact_rows = []
        for _ in range(n_rows):
            row = []
            x = 0
            while len(row) < k:
                x += 1
                full_so_far = [0] + row
                new_diffs = set()
                for a in full_so_far:
                    new_diffs.add(x - a)
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            compact_rows.append(row)
        return compact_rows

    def greedy_lookahead(n_rows, k, rng):
        """
        Greedy with lookahead: when choosing a value, consider how many
        future valid options it leaves for subsequent positions.
        """
        used = set()
        compact_rows = []
        for row_idx in range(n_rows):
            row = []
            x = 0
            while len(row) < k:
                # Find several valid candidates
                valid_candidates = []
                search_limit = x + 60
                for candidate in range(x + 1, search_limit + 1):
                    full_so_far = [0] + row
                    new_diffs = {candidate - a for a in full_so_far}
                    if not new_diffs & used:
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= 10:
                            break
                if not valid_candidates:
                    candidate = x + 1
                    while True:
                        full_so_far = [0] + row
                        new_diffs = {candidate - a for a in full_so_far}
                        if not new_diffs & used:
                            valid_candidates.append(candidate)
                            break
                        candidate += 1

                # Score each candidate by how many options it leaves for next step
                best_score = -1
                best_cand = valid_candidates[0]
                for cand in valid_candidates[:8]:
                    full_so_far = [0] + row
                    new_diffs = {cand - a for a in full_so_far}
                    # Count how many valid next values would exist
                    next_count = 0
                    for next_cand in range(cand + 1, cand + 30):
                        next_full = [0] + row + [cand]
                        next_diffs = {next_cand - a for a in next_full}
                        if not (next_diffs & (used | new_diffs)):
                            next_count += 1
                        if next_count >= 5:
                            break
                    # Prefer smaller values with good future options
                    score = next_count * 100 - cand
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                x = best_cand
                row.append(x)
                full_so_far = [0] + row
                used |= {x - a for a in row}
            compact_rows.append(row)
        return compact_rows

    def greedy_randomized(n_rows, k, rng):
        """Randomized greedy initial solution."""
        used = set()
        compact_rows = []
        for _ in range(n_rows):
            row = []
            x = 0
            while len(row) < k:
                # Find valid candidates
                valid = []
                for candidate in range(x + 1, x + 40):
                    full_so_far = [0] + row
                    new_diffs = {candidate - a for a in full_so_far}
                    if not new_diffs & used:
                        valid.append(candidate)
                        if len(valid) >= 15:
                            break
                if not valid:
                    candidate = x + 1
                    while True:
                        full_so_far = [0] + row
                        new_diffs = {candidate - a for a in full_so_far}
                        if not new_diffs & used:
                            valid.append(candidate)
                            break
                        candidate += 1
                if rng.random() < 0.6:
                    x = valid[0]
                else:
                    x = rng.choice(valid)
                row.append(x)
                full_so_far = [0] + row
                used |= {x - a for a in row}
            compact_rows.append(row)
        return compact_rows

    # Simulated annealing
    def simulate_annealing(initial_rows, n_rows, k, rng, time_budget=100.0):
        """
        SA on compact representation.
        State: list of n_rows compact rows (each of length k).
        Move: pick a random row and a random position, change the value.
        Cost: max entry + penalty * (number of duplicate differences).
        """
        start_time = time.time()
        cur_rows = [list(r) for r in initial_rows]
        cur_max = get_max_entry(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_max = cur_max

        # Compute current cost
        def cost(rows):
            used = set()
            dup_count = 0
            mx = 0
            for cr in rows:
                full = [0] + list(cr)
                for i in range(len(full)):
                    for j in range(i + 1, len(full)):
                        d = full[j] - full[i]
                        if d in used:
                            dup_count += 1
                        else:
                            used.add(d)
                mx = max(mx, max(cr))
            return mx + 100 * dup_count

        cur_cost = cost(cur_rows)
        best_cost = cur_cost

        # Temperature schedule
        T_init = max(50.0, cur_max * 0.5)
        T_final = 0.5
        # We'll run for the time budget
        steps_per_sec_target = 2000  # aim for ~2000 steps/sec

        T = T_init
        step = 0
        no_improve = 0

        while _time_left() > 5.0:
            # Cool down
            T = T_init * (T_final / T_init) ** (step / max(1, int(time_budget * steps_per_sec_target)))
            T = max(T, T_final)

            # Generate neighbor
            row_idx = rng.randrange(n_rows)
            pos_idx = rng.randrange(k)
            cr = cur_rows[row_idx]

            # Determine range for new value
            if pos_idx == 0:
                lo = 1
                hi = cr[0] + 5
            else:
                lo = cr[pos_idx - 1] + 1
                hi = cr[pos_idx] + 5

            if hi <= lo:
                continue

            new_val = rng.randint(lo, hi)
            if new_val == cr[pos_idx]:
                continue

            # Save old value
            old_val = cr[pos_idx]
            cr[pos_idx] = new_val

            # Ensure increasing order (fix if needed)
            if pos_idx > 0 and new_val <= cr[pos_idx - 1]:
                cr[pos_idx] = old_val
                continue
            if pos_idx < k - 1 and new_val >= cr[pos_idx + 1]:
                cr[pos_idx] = old_val
                continue

            new_cost = cost(cur_rows)
            delta = new_cost - cur_cost

            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                # Accept
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in cur_rows]
                    best_max = get_max_entry(best_rows)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Reject
                cr[pos_idx] = old_val

            step += 1

            # Periodic restart if stuck
            if no_improve > 5000 and _time_left() > 20.0:
                # Try a different initial solution
                alt = greedy_randomized(n_rows, k, rng)
                alt_cost = cost(alt)
                if alt_cost < best_cost:
                    best_cost = alt_cost
                    best_rows = [list(r) for r in alt]
                    best_max = get_max_entry(best_rows)
                    cur_rows = [list(r) for r in alt]
                    cur_cost = alt_cost
                    no_improve = 0
                    T = T_init * 0.5

        return best_rows, best_max

    # Local search: try to reduce max entry by adjusting individual values
    def local_search(compact_rows, n_rows, k, rng, max_iters=300):
        """Aggressive local search to reduce max entry."""
        rows = [list(r) for r in compact_rows]
        improved = True
        iters = 0
        while improved and iters < max_iters and _time_left() > 3.0:
            improved = False
            iters += 1
            # Find the row with the largest max entry
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]

            # Try to reduce each element from the end
            for pos in range(k - 1, -1, -1):
                lo = (row[pos - 1] + 1) if pos > 0 else 1
                hi = row[pos] - 1
                if hi < lo:
                    continue
                # Try values from lo upward
                for new_val in range(lo, hi + 1):
                    if new_val == row[pos]:
                        continue
                    # Check validity
                    row[pos] = new_val
                    if is_valid(rows):
                        improved = True
                        break
                    row[pos] = hi  # restore (will be overwritten if we find valid)
                if improved:
                    break

        return rows

    # Multi-pass reduction strategy
    def multi_pass_reduction(compact_rows, n_rows, k, rng, max_rounds=10):
        """
        Systematically try to shrink the largest entries across all rows,
        not just the single maximum row.
        """
        rows = [list(r) for r in compact_rows]
        
        for round_num in range(max_rounds):
            if _time_left() < 5.0:
                break
            
            # Sort rows by their max entry, descending
            row_indices = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
            
            any_improved = False
            for row_idx in row_indices:
                if _time_left() < 5.0:
                    break
                
                row = rows[row_idx]
                # Try to reduce each element from the end
                for pos in range(k - 1, -1, -1):
                    lo = (row[pos - 1] + 1) if pos > 0 else 1
                    hi = row[pos] - 1
                    if hi < lo:
                        continue
                    
                    # Try values from lo upward
                    for new_val in range(lo, hi + 1):
                        if new_val == row[pos]:
                            continue
                        row[pos] = new_val
                        if is_valid(rows):
                            any_improved = True
                            break
                        row[pos] = hi
                    if any_improved:
                        break
            
            if not any_improved:
                break
        
        return rows

    # Main algorithm
    rng = random.Random(42)

    # Generate multiple initial solutions using different strategies
    initial_solutions = []
    initial_solutions.append(greedy_initial(n_rows, k, rng))
    initial_solutions.append(greedy_lookahead(n_rows, k, rng))
    for _ in range(5):
        if _time_left() < 10.0:
            break
        initial_solutions.append(greedy_randomized(n_rows, k, rng))

    # Pick the best initial solution
    best_init = min(initial_solutions, key=lambda r: get_max_entry(r))
    best_max = get_max_entry(best_init)
    best_rows = [list(r) for r in best_init]

    # Run SA on the best initial solution
    if _time_left() > 15.0:
        sa_rows, sa_max = simulate_annealing(best_init, n_rows, k, rng, time_budget=_time_left() - 10.0)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Local search on best
    if _time_left() > 5.0:
        ls_rows = local_search(best_rows, n_rows, k, rng, max_iters=500)
        ls_max = get_max_entry(ls_rows)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Multi-pass reduction
    if _time_left() > 5.0:
        mp_rows = multi_pass_reduction(best_rows, n_rows, k, rng, max_rounds=15)
        mp_max = get_max_entry(mp_rows)
        if mp_max < best_max:
            best_max = mp_max
            best_rows = mp_rows

    # Final validation and conversion
    def validate_full(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    full_rows = [to_full(r) for r in best_rows]
    if not validate_full(full_rows):
        # Fallback
        full_rows = [to_full(r) for r in greedy_initial(n_rows, k, rng)]

    return full_rows
