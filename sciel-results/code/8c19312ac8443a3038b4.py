"""Difference triangle set (4,7): minimize the largest entry."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Total number of differences needed: n_rows * k*(k+1)/2
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # Lower bound on scope: we need total_diffs distinct positive differences,
    # each at most scope. So scope must satisfy scope*(scope+1)/2 >= total_diffs.
    # But also each row has k+1 entries, so the row's largest entry must be at least
    # the largest difference in that row.
    # Simple counting bound: smallest m such that m*(m+1)/2 >= total_diffs
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs:
        lb += 1

    best_scope = None
    best_rows = None

    # We'll try to find a solution with scope close to lb using backtracking.
    # Strategy: assign differences to rows via constraint satisfaction.
    # Each row needs exactly n_diffs_per_row differences from {1, ..., scope}.
    # The differences in a row must be "compatible": there must exist a set of
    # k+1 positions {0 = p_0 < p_1 < ... < p_k} such that {p_j - p_i : i<j}
    # equals the assigned difference set for that row.

    # Alternative approach: directly search for the rows using backtracking
    # on the positions, with pruning based on used differences.

    # For efficiency, we'll use a randomized local search / simulated annealing
    # combined with greedy initialization, since full backtracking may be too slow.

    # First, try a greedy construction to get an initial solution.
    def greedy_construct(n_rows, k, seed=0):
        rng = random.Random(seed)
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

    # Get initial greedy solution
    init_rows = greedy_construct(n_rows, k, seed=42)
    init_scope = max(max(r) for r in init_rows)
    best_scope = init_scope
    best_rows = [list(r) for r in init_rows]

    if time.time() > deadline:
        return best_rows

    # Now try to improve using simulated annealing / local search.
    # Represent a solution as a list of rows.
    # Moves: pick a row, pick a position in that row, try to change it.
    # Or: regenerate one row entirely while keeping others fixed.

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def regenerate_row(row_idx, other_rows, scope_limit, rng):
        """Try to regenerate row_idx given other fixed rows."""
        used = set()
        for r in other_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])

        row = [0]
        x = 0
        max_x = scope_limit
        while len(row) < k + 1:
            x += 1
            if x > max_x:
                return None
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        return row

    # Simulated annealing on row regeneration
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    T = 500.0  # initial temperature
    cooling = 0.9995
    min_T = 1.0

    while time.time() < deadline:
        if T < min_T:
            T = min_T
        improved = False

        # Try to regenerate each row
        for row_idx in range(n_rows):
            other_rows = [current_rows[i] for i in range(n_rows) if i != row_idx]
            # Try to find a row with smaller or equal scope
            new_row = regenerate_row(row_idx, other_rows, current_scope, rng)
            if new_row is not None:
                new_scope = max(current_scope, max(new_row))
                if new_scope < current_scope:
                    current_rows[row_idx] = new_row
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
                    improved = True
                elif new_scope == current_scope and rng.random() < 0.1:
                    # Accept same-scope moves with some probability for diversity
                    current_rows[row_idx] = new_row
                    improved = True

        if not improved:
            # Try with a slightly larger scope to escape local minima
            trial_scope = current_scope + rng.randint(1, 3)
            for row_idx in range(n_rows):
                other_rows = [current_rows[i] for i in range(n_rows) if i != row_idx]
                new_row = regenerate_row(row_idx, other_rows, trial_scope, rng)
                if new_row is not None:
                    new_scope = max(current_scope, max(new_row))
                    delta = new_scope - current_scope
                    if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.1)):
                        current_rows[row_idx] = new_row
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in current_rows]

        T *= cooling

        # Try multiple random seeds for greedy to find better initial solutions
        if rng.random() < 0.05:
            new_init = greedy_construct(n_rows, k, seed=rng.randint(0, 100000))
            new_init_scope = get_scope(new_init)
            if new_init_scope < current_scope:
                current_rows = [list(r) for r in new_init]
                current_scope = new_init_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]

    return best_rows

# EVOLVE-BLOCK-END
