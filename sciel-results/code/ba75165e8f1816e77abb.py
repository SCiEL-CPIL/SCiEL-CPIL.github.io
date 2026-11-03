"""Rewritten solver for Difference triangle set (4,7)."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Build n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # For this specific task: n_rows=4, k=7 -> 8 entries per row, 28 differences per row, 112 total.
    # Best known scope is 134. Counting bound is 112.

    if n_rows <= 0 or k <= 0:
        return []

    # We'll use a randomized local search / simulated annealing approach.
    # State: a list of n_rows lists, each of length k+1, starting at 0.
    # Objective: minimize the maximum element, subject to all differences being unique.

    def valid_differences(rows):
        """Return True if all positive differences are distinct."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def count_conflicts(rows):
        """Return the number of duplicate differences (0 if valid)."""
        diffs = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diffs[d] = diffs.get(d, 0) + 1
        conflicts = 0
        for d, cnt in diffs.items():
            if cnt > 1:
                conflicts += cnt - 1
        return conflicts

    def scope(rows):
        """Return the largest entry across all rows."""
        return max(max(row) for row in rows)

    # --- Strategy 1: Greedy construction with backtracking/randomization ---
    # --- Strategy 2: Local search on a valid or near-valid solution ---

    best_rows = None
    best_scope = float('inf')

    # First, try to get a valid solution via greedy with randomization.
    # Then, try to reduce the scope.

    def greedy_construct(n_rows, k, rng, max_scope_hint=None):
        """
        Greedy construction. Each row is built one element at a time.
        For each new element, we try the smallest possible value that doesn't
        create a duplicate difference with any existing row's differences
        or within the current row.
        """
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            # We want the row to be increasing. We'll pick the next element.
            # To keep scope small, we prefer smaller values.
            # But we must avoid differences already in used_diffs.
            # The new element x creates differences x - row[i] for all i.
            # We need x - row[i] not in used_diffs for all i, and also
            # x - row[i] should be positive (which it is since x > row[i]).
            # Also, within the row, differences must be distinct, but since
            # we're adding one at a time and checking against all previous,
            # and also against global used_diffs, this is covered.
            #
            # Wait, the greedy in the seed program only checks against `used`
            # which includes differences from ALL rows so far. It does NOT
            # explicitly check that differences within the current row are
            # distinct from each other, but since x is increasing and we check
            # x - a for all a in row, and these are all distinct (because a
            # are distinct and x is fixed), the only conflict is with `used`.
            # So the seed's logic is correct for ensuring global uniqueness.
            
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x is too large
                if max_scope_hint is not None and x > max_scope_hint:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    # Try greedy first
    rows = greedy_construct(n_rows, k, random.Random(42))
    if rows is not None:
        if valid_differences(rows):
            best_rows = rows
            best_scope = scope(rows)

    # Now, try to improve the scope using local search.
    # We'll use a simulated annealing-like approach.
    # State: the rows. We'll perturb by changing one element (not the first, which is 0).
    # Cost: conflicts + lambda * (max_scope - target_scope)
    # Actually, let's just minimize conflicts first, then minimize scope.
    # Or use a combined objective.

    # Let's try a different approach: start from the greedy solution,
    # and try to reduce the maximum element by swapping/replacing elements.

    if best_rows is not None:
        current_rows = [list(r) for r in best_rows]
        current_conflicts = count_conflicts(current_rows)
        current_scope = scope(current_rows)
        best_conflicts = current_conflicts
        best_scope_ls = current_scope
        best_rows_ls = [list(r) for r in current_rows]

        T = 1.0  # temperature
        cooling = 0.9995
        rng = random.Random(123)

        max_iter = 500000
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            # Choose a random row and a random position (not 0)
            r = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # position 1 to k (0 is fixed at 0)

            # Try to change current_rows[r][pos] to a nearby value
            old_val = current_rows[r][pos]
            # New value should be greater than current_rows[r][pos-1]
            # and less than current_rows[r][pos+1] if pos < k
            min_val = current_rows[r][pos - 1] + 1
            max_val = (current_rows[r][pos + 1] - 1) if pos < k else current_scope + 10

            # Pick a random new value in a range around old_val
            delta = rng.randint(-20, 20)
            new_val = old_val + delta
            new_val = max(min_val, min(new_val, max_val))

            if new_val == old_val:
                continue

            # Apply the change
            current_rows[r][pos] = new_val

            # Check if the row is still increasing
            if any(current_rows[r][i] >= current_rows[r][i + 1] for i in range(k)):
                current_rows[r][pos] = old_val
                continue

            new_conflicts = count_conflicts(current_rows)
            new_scope = scope(current_rows)

            # Objective: we want to minimize conflicts first, then scope.
            # Let's use a lexicographic objective: (conflicts, scope)
            old_obj = (current_conflicts, current_scope)
            new_obj = (new_conflicts, new_scope)

            # Accept if better, or with some probability if worse
            if new_obj <= old_obj:
                # Accept
                current_conflicts = new_conflicts
                current_scope = new_scope
                if new_obj < (best_conflicts, best_scope_ls):
                    best_conflicts = new_conflicts
                    best_scope_ls = new_scope
                    best_rows_ls = [list(r) for r in current_rows]
            else:
                # Simulated annealing acceptance
                # Delta in objective: we need to define a distance.
                # If conflicts increased, it's a big penalty.
                d_conf = new_conflicts - current_conflicts
                d_scope = new_scope - current_scope
                # Combined delta
                delta_obj = d_conf * 1000 + d_scope
                if delta_obj < 0:
                    accept = True
                else:
                    accept = rng.random() < (1.0 / (1.0 + delta_obj / T))
                
                if accept:
                    current_conflicts = new_conflicts
                    current_scope = new_scope
                    if new_obj < (best_conflicts, best_scope_ls):
                        best_conflicts = new_conflicts
                        best_scope_ls = new_scope
                        best_rows_ls = [list(r) for r in current_rows]
                else:
                    # Revert
                    current_rows[r][pos] = old_val

            T *= cooling

            # If we have a valid solution with a good scope, we can try to reduce T faster
            # or focus on scope reduction.

        # After local search, check if we have a valid solution
        if best_conflicts == 0 and best_scope_ls < best_scope:
            best_rows = best_rows_ls
            best_scope = best_scope_ls

    # If we still don't have a valid solution, try more greedy attempts with different seeds
    if best_rows is None:
        for seed in range(100):
            if time.time() > deadline - 5.0:
                break
            rows = greedy_construct(n_rows, k, random.Random(seed))
            if rows is not None and valid_differences(rows):
                best_rows = rows
                best_scope = scope(rows)
                break

    # If we have a valid solution, try to further optimize scope with a more targeted search
    if best_rows is not None:
        # Try to reduce the maximum element by checking if we can replace the largest element
        # in each row with a smaller one.
        current_rows = [list(r) for r in best_rows]
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for r in range(n_rows):
                # Find the position of the maximum in this row
                max_pos = max(range(len(current_rows[r])), key=lambda i: current_rows[r][i])
                if max_pos == 0:
                    continue
                old_val = current_rows[r][max_pos]
                # Try to reduce old_val
                min_val = current_rows[r][max_pos - 1] + 1
                for new_val in range(old_val - 1, min_val - 1, -1):
                    if new_val < min_val:
                        break
                    current_rows[r][max_pos] = new_val
                    # Check if row is still increasing
                    if any(current_rows[r][i] >= current_rows[r][i + 1] for i in range(k)):
                        continue
                    if valid_differences(current_rows):
                        improved = True
                        break
                    else:
                        current_rows[r][max_pos] = old_val
                        break
                else:
                    current_rows[r][max_pos] = old_val

        if valid_differences(current_rows):
            new_scope = scope(current_rows)
            if new_scope < best_scope:
                best_rows = current_rows
                best_scope = new_scope

    # Final fallback: if no valid solution found, return a trivially valid one (though it may have large scope)
    if best_rows is None:
        # Use the greedy without max_scope_hint, which should always work
        best_rows = greedy_construct(n_rows, k, random.Random(0))
        if best_rows is None:
            # Should not happen, but just in case
            best_rows = [[0]] * n_rows

    return best_rows
# EVOLVE-BLOCK-END
