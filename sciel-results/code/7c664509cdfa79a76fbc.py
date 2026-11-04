"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 143.0
    total_slots = n_rows * k * (k + 1) // 2  # total difference slots

    def max_entry(rows):
        return max(max(row) for row in rows)

    def all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs

    def compute_cost(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        violations = sum(c - 1 for c in counts.values() if c > 1)
        m = max_entry(rows)
        return violations * 10000 + m

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

    # --- Phase 1: Difference-slot greedy coloring initialization ---
    # Each row has k*(k+1)//2 difference slots.
    # A "slot" is identified by (row_idx, i, j) where 0 <= i < j <= k,
    # representing the difference a[j] - a[i] in that row.
    # We want to assign distinct positive integers to all slots across all rows.
    # Constraint: within a row, the differences must be consistent:
    #   a[j] - a[i] = (a[j] - a[m]) + (a[m] - a[i]) for i < m < j
    # So we can't freely assign; we need to assign values a[0]=0, a[1], ..., a[k]
    # such that all pairwise differences are globally unique.
    #
    # Greedy coloring approach:
    # Process slots in order of "impact" (slots involving the largest index first,
    # since they constrain the maximum). Assign the smallest available difference
    # that is consistent with previously assigned values.
    
    def greedy_coloring_init():
        """Initialize using a difference-aware greedy approach.
        We build rows one at a time. For each row, we pick values a[1..k]
        such that all new differences are unused and the row is increasing.
        We use a priority: prefer to keep values small to minimize max entry.
        """
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            # Build the row greedily, choosing smallest valid x at each step
            for pos in range(1, k + 1):
                best_x = None
                # Try from row[-1]+1 upward
                for x in range(row[-1] + 1, 300):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        best_x = x
                        break
                if best_x is None:
                    # Fallback: allow larger values
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            best_x = x
                            break
                if best_x is None:
                    # Should not happen for reasonable parameters
                    best_x = row[-1] + 1
                row.append(best_x)
                used |= {best_x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # --- Alternative init: random restarts with greedy completion ---
    def random_greedy_init(seed_offset=0):
        """Create a random partial assignment and complete greedily."""
        rng = random.Random(seed_offset)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            # Randomly perturb the starting point
            start_bias = rng.randint(0, 5)
            for pos in range(1, k + 1):
                found = False
                # Try a range of candidates
                candidates = list(range(row[-1] + 1 + start_bias, row[-1] + 50 + start_bias))
                rng.shuffle(candidates)
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback: scan upward
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            break
            rows.append(row)
        return rows

    # Try multiple initializations
    best_rows = greedy_coloring_init()
    best_max = max_entry(best_rows)
    best_cost = compute_cost(best_rows)

    # Try a few random inits
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        rows = random_greedy_init(attempt)
        m = max_entry(rows)
        if m < best_max:
            best_max = m
            best_rows = [r[:] for r in rows]
            best_cost = compute_cost(best_rows)

    # --- Phase 2: Constrained backtracking to find valid solution near best_max ---
    def try_backtracking(max_scope, timeout=8.0):
        """Find a valid configuration with all entries <= max_scope."""
        start_time = time.time()
        rows = [[0] for _ in range(n_rows)]

        def dfs(row_idx, pos_idx, used_diffs):
            if time.time() - start_time > timeout:
                return False
            if row_idx == n_rows:
                return True
            if pos_idx == k + 1:
                return dfs(row_idx + 1, 1, used_diffs)

            row = rows[row_idx]
            prev_val = row[pos_idx - 1]

            # Collect valid candidates
            valid_candidates = []
            for x in range(prev_val + 1, max_scope + 1):
                new_diffs = [x - row[i] for i in range(pos_idx)]
                conflict = False
                for d in new_diffs:
                    if d <= 0 or (used_diffs >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)

            if not valid_candidates:
                return False

            # Try in order: smaller first to keep max low, but shuffle slightly
            # Actually, for backtracking to find ANY solution, order matters less.
            # But trying smaller first helps find compact solutions.
            for x in valid_candidates:
                if time.time() - start_time > timeout:
                    return False
                row.append(x)
                new_used = used_diffs
                for i in range(pos_idx):
                    d = x - row[i]
                    new_used |= (1 << d)
                if dfs(row_idx, pos_idx + 1, new_used):
                    return True
                row.pop()
            return False

        if dfs(0, 1, 0):
            return [r[:] for r in rows]
        return None

    # Try backtracking from best_max-1 downward
    for scope in range(best_max - 1, max(112, best_max - 25), -1):
        if time.time() > deadline - 20:
            break
        result = try_backtracking(scope, timeout=6.0)
        if result is not None:
            best_rows = result
            best_max = scope
            best_cost = compute_cost(best_rows)
            # Continue trying lower
            continue
        else:
            # Couldn't find at this scope; stop backtracking
            break

    # --- Phase 3: Simulated Annealing refinement ---
    current_rows = [r[:] for r in best_rows]
    current_cost = compute_cost(current_rows)
    current_max = max_entry(current_rows)

    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]

    T = 30.0
    T_min = 0.005
    cooling = 0.99997

    # Track best
    best_sa_rows = [r[:] for r in current_rows]
    best_sa_max = current_max

    step_count = 0
    while time.time() < deadline:
        if T < T_min:
            # Reheat occasionally
            T = 5.0
            cooling = 0.99997

        r, i = random.choice(indices)
        old_val = current_rows[r][i]

        # Determine valid range
        lower = current_rows[r][i - 1] + 1
        if i < k:
            upper = current_rows[r][i + 1] - 1
        else:
            upper = current_rows[r][k] + 40

        if upper < lower:
            continue

        # Bias: 60% chance to try shrinking (pick from lower half)
        if random.random() < 0.6:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.5)
        else:
            new_val = random.randint(lower, upper)

        if new_val == old_val:
            continue

        current_rows[r][i] = new_val

        # Verify strict increasing
        row = current_rows[r]
        if any(row[j] <= row[j - 1] for j in range(1, len(row))):
            current_rows[r][i] = old_val
            continue

        new_cost = compute_cost(current_rows)
        delta = new_cost - current_cost

        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            new_max = max_entry(current_rows)
            if new_max < best_sa_max:
                best_sa_max = new_max
                best_sa_rows = [rr[:] for rr in current_rows]
        else:
            current_rows[r][i] = old_val

        T *= cooling
        step_count += 1

        # Occasional larger move: rebuild one row
        if step_count % 500 == 0 and time.time() < deadline - 5:
            rebuild_row = random.randint(0, n_rows - 1)
            # Save other rows' used diffs
            other_rows = [current_rows[r2][:] for r2 in range(n_rows) if r2 != rebuild_row]
            other_diffs = set()
            for orow in other_rows:
                for ii in range(len(orow)):
                    for jj in range(ii + 1, len(orow)):
                        other_diffs.add(orow[jj] - orow[ii])
            # Rebuild the row greedily with some randomness
            new_row = [0]
            for pos in range(1, k + 1):
                found = False
                # Try candidates in randomized order
                candidates = list(range(new_row[-1] + 1, new_row[-1] + 80))
                random.shuffle(candidates)
                for x in candidates:
                    new_diffs = {x - a for a in new_row}
                    if not new_diffs & other_diffs:
                        new_row.append(x)
                        found = True
                        break
                if not found:
                    for x in range(new_row[-1] + 1, 400):
                        new_diffs = {x - a for a in new_row}
                        if not new_diffs & other_diffs:
                            new_row.append(x)
                            break
            if len(new_row) == k + 1:
                new_cost_row = compute_cost([new_row] + other_rows)
                delta_row = new_cost_row - current_cost
                if delta_row <= 0 or random.random() < math.exp(-delta_row / T):
                    current_rows[rebuild_row] = new_row
                    current_cost = new_cost_row
                    new_max = max_entry(current_rows)
                    if new_max < best_sa_max:
                        best_sa_max = new_max
                        best_sa_rows = [rr[:] for rr in current_rows]

    # Final check
    if best_sa_max < best_max:
        best_rows = best_sa_rows
        best_max = best_sa_max

    return best_rows
# EVOLVE-BLOCK-END
