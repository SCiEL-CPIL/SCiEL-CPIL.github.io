import time
import random
from collections import defaultdict

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110.0

    # Number of differences per row
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row

    # Heuristic for "small" differences - they are more constrained
    small_threshold = 20

    def count_future_conflicts(used_diffs, row, next_val):
        """
        Count how many differences in the candidate next_val would conflict
        with already used differences, and also estimate how many future
        differences we would block.
        """
        new_diffs = {next_val - a for a in row}
        conflicts = len(new_diffs & used_diffs)
        # Estimate future impact: how many small differences we would use
        small_used = sum(1 for d in new_diffs if d < small_threshold)
        return conflicts * 100 + small_used * 10 + next_val

    def backtrack_row(row, used_diffs, max_val, remaining_slots, depth):
        """
        Try to extend the current row with one more element.
        Returns (success, new_row, new_used_diffs) or (False, None, None).
        """
        if len(row) == k + 1:
            return True, row, used_diffs

        last_val = row[-1]
        best_candidate = None
        best_score = float('inf')
        best_new_diffs = None

        # We need to find x > last_val such that all new differences are unused
        # Also, we should consider the remaining space for subsequent elements
        # A rough upper bound: we need (k+1 - len(row)) - 1 more elements,
        # each at least 1 greater than the previous, so minimum remaining span
        remaining_elements = (k + 1) - len(row)
        min_remaining = remaining_elements * (remaining_elements + 1) // 2

        for x in range(last_val + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue

            # Check if we have enough room for remaining elements
            if x + min_remaining > max_val:
                break

            score = count_future_conflicts(used_diffs, row, x)
            if score < best_score:
                best_score = score
                best_candidate = x
                best_new_diffs = new_diffs

        if best_candidate is None:
            return False, None, None

        new_row = row + [best_candidate]
        new_used = used_diffs | best_new_diffs
        return backtrack_row(new_row, new_used, max_val, remaining_slots, depth + 1)

    def construct_rows_greedy_backtrack(max_scope):
        """
        Try to construct all rows with max entry <= max_scope using backtracking.
        """
        used_diffs = set()
        rows = []

        for _ in range(n_rows):
            # Try to build a row with backtracking
            success, row, _ = backtrack_row(
                [0], used_diffs, max_scope, 0, 0
            )
            if not success:
                return None
            rows.append(row)
            used_diffs |= {row[j] - row[i] for i in range(k + 1) for j in range(i + 1, k + 1)}

        return rows

    # Try different strategies with different time allocations
    best_result = None
    best_scope = float('inf')

    # Strategy 1: Binary search on scope with backtracking
    # Start with a reasonable upper bound
    # Lower bound: we need total_diffs_needed distinct differences, each between 1 and scope
    # So scope must be at least such that 1..scope covers all differences
    # But also each row has k+1 elements, so max element is at least k
    # A tighter bound: the counting bound says scope >= 105 for (7,5)
    low = k  # minimum possible
    high = 150  # start with a generous upper bound

    # First, try a greedy approach to get an initial upper bound
    def greedy_construct(scope_limit):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last_val = row[-1]
                valid = []
                for x in range(last_val + 1, scope_limit + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        # Heuristic: prefer smaller x and fewer small diffs
                        small_count = sum(1 for d in new_diffs if d < small_threshold)
                        valid.append((small_count * 10 + x, x, new_diffs))
                if not valid:
                    return None
                valid.sort(key=lambda t: t[0])
                # Pick the best
                _, x, new_diffs = valid[0]
                row.append(x)
                used_diffs |= new_diffs
            rows.append(row)
        return rows

    # Get initial upper bound from greedy
    greedy_result = greedy_construct(150)
    if greedy_result:
        best_scope = max(r[-1] for r in greedy_result)
        best_result = [r[:] for r in greedy_result]

    # Now try to improve with backtracking, starting from current best_scope - 1
    current_best = best_scope - 1 if best_scope > k else k

    # Try backtracking with decreasing scope
    for scope in range(current_best, low, -1):
        if time.time() - start_time > time_limit:
            break

        result = construct_rows_greedy_backtrack(scope)
        if result:
            current_max = max(r[-1] for r in result)
            if current_max < best_scope:
                best_scope = current_max
                best_result = [r[:] for r in result]

        # If we found a solution at this scope, we can stop
        # because we're searching from high to low
        if result and current_max <= scope:
            break

    # If we still have time, try randomized restarts to improve
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0 and best_result is not None:
        for _ in range(1000):
            if time.time() - start_time > time_limit:
                break

            used_diffs = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    last_val = row[-1]
                    valid_candidates = []
                    for x in range(last_val + 1, best_scope):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            small_count = sum(1 for d in new_diffs if d < small_threshold)
                            valid_candidates.append((small_count, x, new_diffs))
                        if len(valid_candidates) >= 30:
                            break
                    if not valid_candidates:
                        success = False
                        break
                    valid_candidates.sort(key=lambda item: (item[0], item[1]))
                    # Pick from top candidates with some randomness
                    top_n = min(5, len(valid_candidates))
                    chosen = random.choice(valid_candidates[:top_n])
                    row.append(chosen[1])
                    used_diffs |= chosen[2]
                if not success:
                    break
                rows.append(row)

            if success and len(rows) == n_rows:
                current_scope = max(r[-1] for r in rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_result = [r[:] for r in rows]

    if best_result is None:
        # Fallback: deterministic greedy without scope limit
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
