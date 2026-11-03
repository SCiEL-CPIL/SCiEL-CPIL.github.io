"""Difference triangle set (4,7) — optimized backtracking with bitsets."""
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) such that all positive differences
    within rows are distinct across all rows."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Simple counting lower bound on scope:
    # We need total_diffs distinct positive differences, all <= scope.
    # So scope >= total_diffs (since differences 1..scope are the only options).
    # Actually differences can be any value 1..scope, so we need at least total_diffs distinct values in [1, scope].
    lower_bound = total_diffs

    # Upper bound: start with a reasonable estimate. The greedy gives one.
    # Let's first run a fast greedy to get an initial upper bound.
    greedy_rows = _greedy_construct(n_rows, k)
    upper_bound = max(max(r) for r in greedy_rows) if greedy_rows else lower_bound

    # Try to improve via backtracking with decreasing target scope
    best_rows = greedy_rows
    best_scope = upper_bound

    # Try scopes from lower_bound+1 up to best_scope-1 (we want to minimize)
    # But we should try from the lower bound upward, stopping at the first valid.
    # Actually, to minimize scope, we try from lower_bound upward.
    # But that might be too many. Instead, try from best_scope-1 downward? No,
    # we want the smallest valid scope. So try from lower_bound upward.

    # However, trying every scope from lower_bound up could be expensive.
    # Strategy: try a few scopes near the lower bound, and also do a final
    # polish pass trying to reduce by 1 from the best found.

    # First, attempt backtracking for scopes in a targeted range.
    # We'll try scopes from max(lower_bound, best_scope - 10) up to best_scope - 1,
    # but also include the lower bound itself if it's not too far.

    # Actually, let's try from lower_bound upward, but with a time budget per scope.
    # If a scope fails quickly, move on. If it's near the lower bound, give it more time.

    # Better: try scopes in order from lower_bound to best_scope - 1.
    # For each, run backtracking with a time limit. If found, that's optimal
    # (since we're going upward). If not found within time, move to next.

    # But the search space is huge. Let's be smart:
    # 1. Try the lower bound first with a generous time limit.
    # 2. Then try lower_bound + 1, +2, ... up to best_scope - 1.
    # 3. Stop at the first scope that yields a valid construction.

    # Time allocation: give more time to lower scopes.
    # Total time budget for search: deadline - start_time - 10s (safety)
    # Allocate proportionally.

    # Let's try a different approach: binary search is not directly applicable
    # because validity is not monotonic in a simple way... actually it IS monotonic:
    # if a valid set exists with scope S, then a valid set exists with scope S+1
    # (just keep the same set, the max entry is still <= S+1). Wait, no: the scope
    # is the largest entry. If we have a valid set with max entry S, we can always
    # use it for any scope >= S. So validity is monotonic: if scope S is feasible,
    # all S' >= S are feasible. We want the minimum feasible S.

    # So binary search on S in [lower_bound, best_scope] is valid!
    # But backtracking for a given S may be very hard. Let's do a linear scan
    # from lower_bound upward, with increasing time budgets.

    # Actually, let's just do a linear scan from lower_bound to best_scope-1.
    # For each candidate scope, run backtracking with a time limit.
    # The time limit increases for higher scopes (since they're more likely to be feasible).

    # But this could be slow. Let's try a hybrid:
    # - First, try lower_bound with a moderate time limit.
    # - If that fails, try lower_bound+1, etc.
    # - If we run out of time, return the best found.

    # Let's implement the backtracking search.

    # We'll try scopes from lower_bound up to best_scope - 1.
    # For each scope, we have a time budget.
    # Total search time available: deadline - start_time - 5 (safety)

    remaining_time = deadline - start_time - 5.0
    if remaining_time <= 1:
        return best_rows

    # Number of scopes to try
    num_scopes_to_try = best_scope - lower_bound
    if num_scopes_to_try <= 0:
        return best_rows

    # Distribute time: give more time to lower scopes (harder to find)
    # Use a simple heuristic: 70% of time for the first scope, rest split.
    # Actually, let's just give each scope an equal share, but cap the total.
    time_per_scope = remaining_time / max(1, min(num_scopes_to_try, 20))
    # Cap at 30 seconds per scope
    time_per_scope = min(time_per_scope, 30.0)

    for scope in range(lower_bound, best_scope):
        if time.time() > deadline - 5.0:
            break

        # Time budget for this scope
        t_start = time.time()
        t_budget = min(time_per_scope, deadline - t_start - 5.0)
        if t_budget <= 0.5:
            break

        rows = _backtrack_construct(n_rows, k, scope, t_budget, start_time)
        if rows is not None:
            best_rows = rows
            best_scope = scope
            # Since we're going upward, this is the minimum feasible scope.
            # We can stop.
            break

    # Final polish: try to reduce scope by 1 from best_scope
    if time.time() < deadline - 10.0 and best_scope > lower_bound:
        t_start = time.time()
        t_budget = min(20.0, deadline - t_start - 5.0)
        if t_budget > 1.0:
            rows = _backtrack_construct(n_rows, k, best_scope - 1, t_budget, start_time)
            if rows is not None:
                best_rows = rows
                best_scope = best_scope - 1

    return best_rows


def _greedy_construct(n_rows, k):
    """Fast greedy construction to get an initial upper bound."""
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


def _backtrack_construct(n_rows, k, max_scope, time_budget, start_time):
    """Backtracking search for a valid construction with max entry <= max_scope.
    Returns a list of rows if found, else None."""
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # We need total_diffs distinct differences, all in [1, max_scope].
    # So we need max_scope >= total_diffs.
    if max_scope < total_diffs:
        return None

    # Use a bitset for tracking used differences.
    # max_scope can be up to ~200, so we use a Python int as a bitset.
    used_mask = 0  # bit d is set if difference d is used

    rows = []

    def backtrack(row_idx, row, row_used_mask):
        """Try to construct row row_idx, given the current row and its used mask."""
        if time.time() - start_time > time_budget:
            return False

        if row_idx == n_rows:
            return True

        if len(row) == k + 1:
            # Row complete, move to next row
            rows.append(row)
            success = backtrack(row_idx + 1, [0], 0)
            if success:
                return True
            rows.pop()
            return False

        # Try next values for the last element of the row
        last = row[-1] if row else 0
        # The next value must be > last and <= max_scope
        # Also, the new differences must not conflict with used_mask or row_used_mask
        for x in range(last + 1, max_scope + 1):
            # Check if x <= max_scope
            # Compute new differences
            new_diffs = []
            conflict = False
            for a in row:
                d = x - a
                # d is in [1, max_scope]
                bit = 1 << d
                if (used_mask | row_used_mask) & bit:
                    conflict = True
                    break
                new_diffs.append(d)
            if conflict:
                continue

            # Check if all new_diffs are distinct (they should be since row is increasing)
            # They are distinct because a is different for each element.

            # Place x
            row.append(x)
            new_row_mask = 0
            for d in new_diffs:
                new_row_mask |= (1 << d)

            if backtrack(row_idx, row, row_used_mask | new_row_mask):
                return True

            row.pop()

        return False

    # Start backtracking
    success = backtrack(0, [0], 0)
    if success:
        return rows
    return None


# EVOLVE-BLOCK-END
