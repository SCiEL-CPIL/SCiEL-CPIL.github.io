"""
Strategy: Iterated Local Search (ILS) with Fast Delta-Cost Evaluation.

The core structural change is the move operator and the cost evaluation:
1. Fast delta-cost: When moving a single element, only the O(k) differences
   that involve that element change. We maintain a global frequency array of
   all differences and compute the exact delta in O(k) instead of O(n_rows * k^2).
2. ILS with perturbation: Instead of pure SA with cooling, we use a restart
   strategy. After a local optimum is found, we perturb the solution and
   search again. This avoids getting stuck in local minima.
3. Binary search on scope: We binary-search the maximum allowed scope.
   For each candidate scope, we run ILS to find a valid solution within that scope.
   This directly targets the objective (minimize max entry).
"""
import random
import time
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Helper to validate
    def is_valid(rows, max_scope):
        used = set()
        for row in rows:
            if max(row) > max_scope:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Greedy construction: build rows one by one, each row picks smallest
    # available values that don't conflict with existing differences
    def greedy_build(max_scope):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                # Check if x can be added
                new_diffs = {x - a for a in row}
                if not new_diffs & used and x <= max_scope:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows

    # Get initial greedy solution
    greedy_rows = greedy_build(200)
    best_max = max(max(row) for row in greedy_rows)
    best_rows = [row[:] for row in greedy_rows]

    # Fast delta-cost ILS
    def ils_search(max_scope, time_budget=30.0):
        """
        Iterated Local Search to find a valid solution with all entries <= max_scope.
        Returns (rows, is_valid) tuple.
        """
        start = time.time()
        end = start + time_budget

        # Build difference frequency array
        # Max possible diff is max_scope
        max_diff = max_scope
        diff_count = [0] * (max_diff + 1)  # 1-indexed, diff_count[d] = count of difference d
        diff_owners = {}  # diff -> list of (row_idx, i, j) where row[j]-row[i]==d

        def build_diff_struct(rows):
            nonlocal diff_count, diff_owners
            diff_count = [0] * (max_diff + 1)
            diff_owners = {}
            for r in range(len(rows)):
                row = rows[r]
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        diff_count[d] += 1
                        if d not in diff_owners:
                            diff_owners[d] = []
                        diff_owners[d].append((r, i, j))

        def num_conflicts():
            c = 0
            for d in range(1, len(diff_count)):
                if diff_count[d] > 1:
                    c += diff_count[d] - 1
            return c

        def apply_move(rows, r, c_idx, old_val, new_val):
            """Update diff structures for a move. Returns delta_conflicts."""
            nonlocal diff_count, diff_owners
            delta = 0
            row = rows[r]
            # Remove old differences involving position c_idx
            for j in range(len(row)):
                if j == c_idx:
                    continue
                if j > c_idx:
                    d = row[j] - old_val
                else:
                    d = old_val - row[j]
                diff_count[d] -= 1
                delta -= 1  # temporary, we'll recompute properly
                # Remove from owners
                if d in diff_owners:
                    lst = diff_owners[d]
                    for idx, (rr, ii, jj) in enumerate(lst):
                        if rr == r and ii == j and jj == c_idx:
                            lst.pop(idx)
                            break
                        elif rr == r and ii == c_idx and jj == j:
                            lst.pop(idx)
                            break
                if diff_count[d] == 0:
                    del diff_owners[d]
                # Update delta
                if diff_count[d] >= 1:
                    delta += 1  # still has at least one, so one fewer conflict if it was >1
                # Actually let's just recompute conflicts from scratch for safety
                # No, that's too slow. Let me think differently.
            # This approach is getting complicated. Let me use a simpler method.
            # Just recompute total conflicts after the move.
            return 0  # placeholder

        # Simpler approach: maintain total_conflicts and update incrementally
        def compute_total_conflicts():
            t = 0
            for d in range(1, len(diff_count)):
                if diff_count[d] > 1:
                    t += diff_count[d] - 1
            return t

        def move_element(rows, r, c_idx, new_val):
            """
            Move element at rows[r][c_idx] to new_val.
            Update diff_count and return the change in number of conflicts.
            """
            nonlocal diff_count
            row = rows[r]
            old_val = row[c_idx]
            if old_val == new_val:
                return 0

            # Before: count conflicts involving old_val
            old_conflicts = 0
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d_old = abs(row[j] - old_val)
                old_conflicts += max(0, diff_count[d_old] - 1)

            # Remove old differences
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d = abs(row[j] - old_val)
                diff_count[d] -= 1

            # After removing, count conflicts
            mid_conflicts = 0
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d = abs(row[j] - new_val)
                mid_conflicts += max(0, diff_count[d])

            # Add new differences
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d = abs(row[j] - new_val)
                diff_count[d] += 1

            # After adding, count conflicts
            new_conflicts = 0
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d = abs(row[j] - new_val)
                new_conflicts += max(0, diff_count[d] - 1)

            # Net change: new_conflicts - old_conflicts
            # But this double-counts. Let me just compute delta as:
            # delta = (conflicts_after - conflicts_before)
            # conflicts_before = sum over all d of max(0, diff_count[d]-1) before the move
            # This is expensive to compute each time.

            # Alternative: just track total_conflicts globally
            return 0  # We'll use a different approach

        # Let me use the simplest correct approach:
        # Maintain diff_count array and total_conflicts.
        # When moving an element, compute old and new contributions.

        def make_move(rows, r, c_idx, new_val, diff_count, total_conflicts):
            """
            Returns (new_total_conflicts, success) where success means new_val is valid
            (within bounds, maintains increasing order).
            """
            row = rows[r]
            old_val = row[c_idx]
            if old_val == new_val:
                return total_conflicts, True

            # Check bounds
            if new_val < row[c_idx - 1] + 1:
                return total_conflicts, False
            if c_idx < len(row) - 1 and new_val > row[c_idx + 1] - 1:
                return total_conflicts, False
            if new_val > max_scope:
                return total_conflicts, False

            # Compute change in conflicts
            # For each j != c_idx, the difference |row[j] - old_val] changes to |row[j] - new_val]
            delta_conflicts = 0
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d_old = abs(row[j] - old_val)
                d_new = abs(row[j] - new_val)

                # Before: d_old had count diff_count[d_old]
                # After removing: d_old has count diff_count[d_old] - 1
                # Conflict contribution of d_old before: max(0, diff_count[d_old] - 1)
                # Conflict contribution of d_old after: max(0, diff_count[d_old] - 1 - 1) = max(0, diff_count[d_old] - 2)

                # Before: d_new had count diff_count[d_new]
                # After adding: d_new has count diff_count[d_new] + 1
                # Conflict contribution of d_new before: max(0, diff_count[d_new] - 1)
                # Conflict contribution of d_new after: max(0, diff_count[d_new] + 1 - 1) = max(0, diff_count[d_new])

                c_old_before = max(0, diff_count[d_old] - 1)
                c_old_after = max(0, diff_count[d_old] - 2)
                c_new_before = max(0, diff_count[d_new] - 1)
                c_new_after = max(0, diff_count[d_new])

                delta_conflicts += (c_old_after + c_new_after) - (c_old_before + c_new_before)

            # Update diff_count
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d_old = abs(row[j] - old_val)
                d_new = abs(row[j] - new_val)
                diff_count[d_old] -= 1
                diff_count[d_new] += 1

            # Apply the move
            row[c_idx] = new_val

            return total_conflicts + delta_conflicts, True

        def undo_move(rows, r, c_idx, new_val, old_val, diff_count, total_conflicts):
            """Undo a move."""
            row = rows[r]
            delta_conflicts = 0
            for j in range(len(row)):
                if j == c_idx:
                    continue
                d_new = abs(row[j] - new_val)
                d_old = abs(row[j] - old_val)

                c_new_before = max(0, diff_count[d_new] - 1)
                c_new_after = max(0, diff_count[d_new] - 2)
                c_old_before = max(0, diff_count[d_old] - 1)
                c_old_after = max(0, diff_count[d_old])

                delta_conflicts += (c_old_after + c_new_after) - (c_old_before + c_new_before)

            for j in range(len(row)):
                if j == c_idx:
                    continue
                d_new = abs(row[j] - new_val)
                d_old = abs(row[j] - old_val)
                diff_count[d_new] -= 1
                diff_count[d_old] += 1

            row[c_idx] = old_val
            return total_conflicts + delta_conflicts

        # ILS main loop
        # Start from a valid solution if possible, else random
        current_rows = [row[:] for row in greedy_rows]
        # Clamp to max_scope
        for row in current_rows:
            for i in range(len(row)):
                row[i] = min(row[i], max_scope)
            # Ensure strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    row[i] = row[i-1] + 1
            if row[-1] > max_scope:
                # Need to compress
                # This is problematic; just scale down
                scale = max_scope / max(row[-1], 1)
                row = [0] + sorted(set(int(x * scale) for x in row[1:]))
                # Ensure length
                while len(row) < k + 1:
                    row.append(row[-1] + 1)
                row = row[:k+1]
                for i in range(1, len(row)):
                    if row[i] <= row[i-1]:
                        row[i] = row[i-1] + 1
                current_rows[0] = row  # This is wrong, we need per-row

        # Let me restart this more carefully
        def random_solution(max_scope):
            rows = []
            for _ in range(n_rows):
                # Pick k distinct values from 1..max_scope
                if max_scope >= k:
                    vals = sorted(random.sample(range(1, max_scope + 1), k))
                    row = [0] + vals
                else:
                    row = list(range(k + 1))
                rows.append(row)
            return rows

        # Try to find valid solution
        # Start with greedy, check if valid
        current_rows = greedy_build(max_scope)
        if max(max(row) for row in current_rows) > max_scope:
            # Greedy exceeded scope, use random
            current_rows = random_solution(max_scope)

        # Build diff_count
        diff_count = [0] * (max_scope + 1)
        for row in current_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_scope:
                        diff_count[d] += 1

        total_conflicts = 0
        for d in range(1, max_scope + 1):
            if diff_count[d] > 1:
                total_conflicts += diff_count[d] - 1

        # If already valid, return
        if total_conflicts == 0 and max(max(row) for row in current_rows) <= max_scope:
            return current_rows, True

        # ILS with local search
        best_found = None
        best_conflicts = total_conflicts

        iteration = 0
        while time.time() < end:
            # Local search: try to reduce conflicts
            improved = True
            max_iters = 500
            iters = 0
            while improved and iters < max_iters and time.time() < end:
                improved = False
                iters += 1
                # Pick random row and position
                r = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, k)

                # Try to find a better value
                row = current_rows[r]
                old_val = row[c_idx]

                # Candidate values: try a few random values in valid range
                min_v = row[c_idx - 1] + 1
                max_v = min(max_scope, row[c_idx + 1] - 1) if c_idx < k else max_scope

                if min_v > max_v:
                    continue

                # Try up to 5 random candidate values
                tried = set()
                for _ in range(5):
                    new_val = random.randint(min_v, max_v)
                    if new_val in tried or new_val == old_val:
                        continue
                    tried.add(new_val)

                    new_tc, ok = make_move(current_rows, r, c_idx, new_val, diff_count, total_conflicts)
                    if not ok:
                        continue

                    if new_tc < total_conflicts:
                        total_conflicts = new_tc
                        improved = True
                        if total_conflicts < best_conflicts:
                            best_conflicts = total_conflicts
                            best_found = [row2[:] for row2 in current_rows]
                        if total_conflicts == 0:
                            if max(max(row2) for row2 in current_rows) <= max_scope:
                                return current_rows, True
                        break
                    else:
                        # Undo
                        total_conflicts = undo_move(current_rows, r, c_idx, new_val, old_val, diff_count, total_conflicts)

            # If local search stuck, perturb
            if best_conflicts == 0 and best_found:
                if max(max(row) for row in best_found) <= max_scope:
                    return best_found, True
                # Try to fix scope issue
                # Scale down
                for row in best_found:
                    if row[-1] > max_scope:
                        # Compress
                        scale = max_scope / row[-1]
                        new_row = [0] + sorted(set(int(x * scale) for x in row[1:]))
                        while len(new_row) < k + 1:
                            new_row.append(new_row[-1] + 1)
                        new_row = new_row[:k+1]
                        for i in range(1, len(new_row)):
                            if new_row[i] <= new_row[i-1]:
                                new_row[i] = new_row[i-1] + 1
                        best_found[best_found.index(row)] = new_row
                # Rebuild diff_count
                diff_count = [0] * (max_scope + 1)
                for row in best_found:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d <= max_scope:
                                diff_count[d] += 1
                total_conflicts = 0
                for d in range(1, max_scope + 1):
                    if diff_count[d] > 1:
                        total_conflicts += diff_count[d] - 1
                current_rows = [row[:] for row in best_found]
                best_conflicts = total_conflicts
                if total_conflicts == 0 and max(max(row) for row in current_rows) <= max_scope:
                    return current_rows, True

            # Perturb: make a few random moves that may increase conflicts
            for _ in range(3):
                r = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, k)
                row = current_rows[r]
                old_val = row[c_idx]
                min_v = row[c_idx - 1] + 1
                max_v = min(max_scope, row[c_idx + 1] - 1) if c_idx < k else max_scope
                if min_v <= max_v:
                    new_val = random.randint(min_v, max_v)
                    if new_val != old_val:
                        new_tc, ok = make_move(current_rows, r, c_idx, new_val, diff_count, total_conflicts)
                        if ok:
                            total_conflicts = new_tc

            iteration += 1

        # Return best found
        if best_found and best_conflicts == 0:
            if max(max(row) for row in best_found) <= max_scope:
                return best_found, True

        # Check current
        if total_conflicts == 0 and max(max(row) for row in current_rows) <= max_scope:
            return current_rows, True

        return None, False

    # Binary search on scope
    low = total_diffs  # lower bound
    high = best_max

    # First, make sure high is feasible
    while low < high and time.time() < deadline - 5:
        mid = (low + high) // 2
        # Try to find solution with scope mid
        rows, valid = ils_search(mid, time_budget=min(20.0, max(5.0, deadline - time.time() - 5)))
        if valid:
            high = mid
            best_max = mid
            best_rows = rows
        else:
            low = mid + 1

    # Final: try to improve best_max by 1, 2, ...
    for target in range(best_max - 1, max(low - 1, best_max - 50) - 1, -1):
        if time.time() > deadline - 3:
            break
        rows, valid = ils_search(target, time_budget=min(15.0, max(3.0, deadline - time.time() - 3)))
        if valid:
            best_max = target
            best_rows = rows

    # Ensure we return a valid solution
    if best_rows is None:
        best_rows = greedy_build(best_max)

    # Final validation
    if not is_valid(best_rows, max(max(row) for row in best_rows)):
        # Fallback to greedy
        greedy = greedy_build(200)
        best_rows = greedy
        best_max = max(max(row) for row in greedy)

    return best_rows
