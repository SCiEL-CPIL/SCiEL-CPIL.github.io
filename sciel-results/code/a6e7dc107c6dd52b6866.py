"""Difference triangle set construction with interleaved greedy + SA polish."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(12345)

    # --- Helper: compute all positive differences for a row ---
    def row_diffs(row):
        d = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d.add(row[j] - row[i])
        return d

    # --- Interleaved greedy initialization ---
    # Build all rows simultaneously, one element at a time, to balance the used-diff space.
    def interleaved_greedy(n_rows, k, seed):
        rng_local = random.Random(seed)
        rows = [[0] for _ in range(n_rows)]
        used = set()
        # We need to add k elements to each row (positions 1..k)
        # Interleave: for each "level" t in 1..k, add element t to each row
        for t in range(1, k + 1):
            # Randomize row order for this level to break symmetry
            order = list(range(n_rows))
            rng_local.shuffle(order)
            for r in order:
                row = rows[r]
                last = row[-1]
                # Try increasing x starting from last+1
                x = last + 1
                max_x = 200  # hard cap per element
                while x <= max_x:
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        break
                    x += 1
                else:
                    # Fallback: if we hit max_x, just take the next available
                    # (shouldn't happen for reasonable params)
                    row.append(x)
                    used |= {x - a for a in row}
        return rows

    # --- Scope (largest entry) ---
    def scope(rows):
        return max(max(r) for r in rows)

    # --- Validity check ---
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            for d in row_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    # --- Simulated Annealing polish ---
    def sa_polish(rows, n_rows, k, time_budget):
        if not is_valid(rows):
            return rows
        best = [list(r) for r in rows]
        best_scope = scope(best)
        current = [list(r) for r in best]
        current_scope = best_scope

        # Precompute used diffs for current
        def compute_used(rows_):
            s = set()
            for r in rows_:
                for d in row_diffs(r):
                    s.add(d)
            return s

        used = compute_used(current)

        # SA parameters
        T = 30.0
        T_min = 0.1
        cooling = 0.9995

        start_time = time.time()
        steps = 0
        while time.time() < start_time + time_budget:
            steps += 1
            # Pick a random row and a random position (not 0)
            r_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)

            old_val = current[r_idx][pos]
            # Try to move this element: shift by delta
            # Bias towards reducing scope: try smaller values first
            if current_scope > 105:
                # Try to reduce
                deltas = []
                # Try decreasing
                for d in [1, 2, 3, 5, 8, 13, 21, 34]:
                    new_val = old_val - d
                    if new_val > current[r_idx][pos - 1]:
                        deltas.append(new_val)
                # Also try increasing slightly (to escape local minima)
                for d in [1, 2, 3, 5]:
                    deltas.append(old_val + d)
                # Random
                for _ in range(3):
                    deltas.append(old_val + rng.randint(-10, 10))
            else:
                deltas = [old_val + rng.randint(-5, 5) for _ in range(5)]

            rng.shuffle(deltas)
            moved = False
            for new_val in deltas:
                # Check bounds: must be > previous and < next (if exists)
                if new_val <= current[r_idx][pos - 1]:
                    continue
                if pos < k and new_val >= current[r_idx][pos + 1]:
                    continue

                # Compute old diffs for this row
                old_row = current[r_idx][:]
                old_diffs = row_diffs(old_row)

                # Temporarily modify
                current[r_idx][pos] = new_val
                new_row = current[r_idx]
                new_diffs = row_diffs(new_row)

                # Check for conflicts: new_diffs must not intersect with (used - old_diffs)
                other_diffs = used - old_diffs
                if new_diffs & other_diffs:
                    current[r_idx][pos] = old_val
                    continue

                # Valid move! Compute new scope
                new_scope = scope(current)
                delta_scope = new_scope - current_scope

                # Accept?
                if delta_scope <= 0:
                    accept = True
                else:
                    accept = rng.random() < _exp_neg(delta_scope, T)

                if accept:
                    # Update used
                    used = (used - old_diffs) | new_diffs
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [list(r) for r in current]
                    moved = True
                    break
                else:
                    current[r_idx][pos] = old_val

            if not moved:
                pass  # no move accepted

            T *= cooling
            if T < T_min:
                T = T_min

        return best

    def _exp_neg(x, T):
        if T <= 0:
            return 0.0
        if x / T > 700:
            return 0.0
        return 2.718281828459045 ** (-x / T)

    # --- Main search loop ---
    # Try multiple seeds for interleaved greedy, then SA polish the best
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Generate several interleaved greedy solutions
    num_initial = 8
    candidates = []
    for i in range(num_initial):
        rows = interleaved_greedy(n_rows, k, seed=1000 + i * 7)
        if is_valid(rows):
            s = scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
        if time.time() > deadline - 60:
            break

    # Sort by scope
    candidates.sort(key=lambda x: x[0])

    # Phase 2: SA polish on the best candidates
    # Allocate remaining time to SA
    remaining = deadline - time.time() - 5.0
    if remaining > 2:
        # Polish the top 3 candidates
        top_n = min(3, len(candidates))
        time_per = max(1.0, remaining / top_n)
        for i in range(top_n):
            if time.time() > deadline - 5:
                break
            s, rows = candidates[i]
            polished = sa_polish(rows, n_rows, k, time_budget=time_per)
            if is_valid(polished):
                ps = scope(polished)
                if ps < best_scope:
                    best_scope = ps
                    best_rows = [list(r) for r in polished]

    # Fallback: if no valid solution found, use sequential greedy
    if best_rows is None:
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
        best_rows = rows

    return best_rows
# EVOLVE-BLOCK-END
