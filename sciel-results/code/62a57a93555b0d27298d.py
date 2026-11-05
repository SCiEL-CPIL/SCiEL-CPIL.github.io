import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0

    # Total distinct differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound: need at least total_diffs distinct positive integers
    # The max entry must be at least total_diffs (since diffs are in [1, max_entry])
    lower_bound = total_diffs

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_diffs(row):
        """Return set of positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_build_row(fixed_diffs, max_limit, seed_val):
        """
        Build a single row of k+1 increasing integers starting at 0,
        avoiding all differences in fixed_diffs, with max entry <= max_limit.
        Uses greedy: pick smallest possible next value that doesn't create
        a conflict with fixed_diffs or previously chosen values in this row.
        Returns None if stuck before completing k+1 entries.
        """
        rng = random.Random(seed_val)
        row = [0]
        used_in_row = set()  # differences created within this row so far
        
        for _ in range(k):
            # Find smallest x > row[-1] such that:
            # 1. x - row[i] not in fixed_diffs for all i
            # 2. x - row[i] not in used_in_row for all i
            # 3. x <= max_limit
            found = False
            # Try values in a slightly randomized order near the minimum to break ties
            min_candidate = row[-1] + 1
            # Search up to max_limit
            # For efficiency, check sequentially but with occasional randomization
            search_start = min_candidate
            # We'll check sequentially for determinism within a seed
            x = search_start
            while x <= max_limit:
                valid = True
                for a in row:
                    d = x - a
                    if d in fixed_diffs:
                        valid = False
                        break
                    if d in used_in_row:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    # Update used_in_row with new differences
                    for a in row[:-1]:
                        used_in_row.add(x - a)
                    found = True
                    break
                x += 1
            if not found:
                return None
        return row

    def greedy_build_all(max_limit, seed_base):
        """Build all rows greedily, each avoiding the fixed diffs of previous rows."""
        rows = []
        fixed_diffs = set()
        for i in range(n_rows):
            row = greedy_build_row(fixed_diffs, max_limit, seed_base + i)
            if row is None:
                return None
            rows.append(row)
            fixed_diffs |= compute_diffs(row)
        return rows

    def repair_row(rows, row_idx, max_limit, seed_val):
        """
        Rebuild row row_idx, avoiding all differences from other rows.
        Returns new row or None if failed.
        """
        # Collect fixed diffs from all other rows
        fixed_diffs = set()
        for i, row in enumerate(rows):
            if i == row_idx:
                continue
            fixed_diffs |= compute_diffs(row)
        
        new_row = greedy_build_row(fixed_diffs, max_limit, seed_val)
        if new_row is None:
            return None
        return new_row

    def local_tighten(rows, max_limit, seed_base):
        """
        Try to reduce the max entry by tightening individual elements.
        For each row, try to rebuild it with a smaller max_limit.
        Also try swapping elements between rows if it reduces max.
        """
        current_max = get_max_entry(rows)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to tighten each row
            for r in range(n_rows):
                if time.time() > deadline:
                    break
                # What is the current max of this row?
                row_max = max(rows[r])
                # Try to rebuild this row with a smaller bound
                # The bound for this row alone can be row_max - 1, but we need
                # to ensure it doesn't conflict with others.
                # Actually, we can try to rebuild with bound = current_max - 1
                # but that's global. Let's try to rebuild with bound = row_max - 1
                # if row_max > lower_bound // n_rows (rough heuristic)
                # Better: try to rebuild with bound = min(row_max - 1, max_limit - 1)
                new_bound = row_max - 1
                if new_bound < k:  # Must be at least k to fit k+1 distinct ints starting at 0
                    continue
                new_row = repair_row(rows, r, new_bound, seed_base + r * 1000)
                if new_row is not None:
                    # Check if it actually reduces the global max
                    old_global_max = get_max_entry(rows)
                    rows[r] = new_row
                    new_global_max = get_max_entry(rows)
                    if new_global_max < old_global_max:
                        max_limit = new_global_max
                        improved = True
                    else:
                        # Revert if no improvement in global max
                        # Actually, we want to keep it if it's valid and same max?
                        # No, we want to minimize max. If max didn't drop, revert to keep search focused.
                        # But maybe it opens up space for other rows?
                        # For now, only accept if global max drops.
                        # To allow exploration, we could accept if valid and max same, but let's be strict first.
                        pass
        return rows

    # Main loop
    best_rows = None
    best_max = float('inf')

    # Initial greedy construction
    # Start with a generous bound, then tighten
    initial_bound = lower_bound + n_rows * k + 10
    for attempt in range(10):
        if time.time() > deadline:
            break
        rows = greedy_build_all(initial_bound, attempt * 12345)
        if rows is not None:
            curr_max = get_max_entry(rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [row[:] for row in rows]
            # Try to tighten
            rows = local_tighten(rows, best_max - 1, attempt * 12345 + 1)
            curr_max = get_max_entry(rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [row[:] for row in rows]

    # If we haven't found a good solution, try a different initialization
    if best_rows is None:
        # Fallback: pure greedy without max limit
        rows = []
        fixed_diffs = set()
        for i in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & fixed_diffs:
                    row.append(x)
                    fixed_diffs |= new_diffs
            rows.append(row)
        best_rows = rows
        best_max = get_max_entry(best_rows)

    # Now, run a series of repair-based local searches
    # Strategy: Pick a random row, rebuild it with a tight bound, see if max drops.
    # Also try "row swaps": rebuild two rows simultaneously? Too complex.
    # Let's stick to single-row repair with varying bounds.

    num_repairs = 200
    for i in range(num_repairs):
        if time.time() > deadline:
            break
        
        # Work on a copy of the best solution
        current_rows = [row[:] for row in best_rows]
        current_max = best_max
        
        # Pick a random row to repair
        r = random.randint(0, n_rows - 1)
        
        # Try to rebuild this row with a bound slightly below current_max
        # The bound should be at least the row's own max, but we want to reduce it.
        # Let's try bound = current_max - 1
        # If that fails, try current_max - 2, etc., down to a reasonable limit.
        # But rebuilding with a smaller bound might fail.
        
        # Alternative: Rebuild with bound = current_max, but with a different seed,
        # hoping to find a configuration with a smaller max entry naturally.
        # This is less targeted.
        
        # Let's try: Rebuild row r with bound = current_max - 1.
        # If successful and valid (which it is by construction), check if global max drops.
        # Note: The rebuilt row might still have max = current_max - 1, but other rows
        # might have max = current_max. So global max stays current_max unless the
        # rebuilt row was the unique holder of current_max AND its new max is smaller.
        
        # To force a drop, we need to rebuild the row that holds the max.
        # Let's identify which rows hold the current max.
        max_rows = [i for i in range(n_rows) if max(current_rows[i]) == current_max]
        
        if not max_rows:
            continue
            
        # Pick one of the max-holding rows
        target_row = random.choice(max_rows)
        
        # Try to rebuild it with a smaller bound
        # The new bound must be < current_max
        # Also, the new row must fit within the bound
        new_bound = current_max - 1
        if new_bound < k:
            continue
            
        new_row = repair_row(current_rows, target_row, new_bound, i * 777 + target_row)
        if new_row is not None:
            current_rows[target_row] = new_row
            new_global_max = get_max_entry(current_rows)
            if new_global_max < current_max:
                best_max = new_global_max
                best_rows = [row[:] for row in current_rows]

    # Final polish: try to tighten all rows with the new best_max
    if best_rows is not None:
        best_rows = local_tighten(best_rows, best_max - 1, 999999)
        best_max = get_max_entry(best_rows)

    return best_rows
