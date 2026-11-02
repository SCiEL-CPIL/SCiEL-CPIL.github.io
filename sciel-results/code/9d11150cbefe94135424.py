"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 110  # 110s budget, leaving 10s safety margin
    
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Counting bound: minimum possible scope m such that m*(m+1)/2 >= total_diffs
    # Solve m^2 + m - 2*total_diffs >= 0
    import math
    m_min = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    # Start with greedy to get an initial solution
    best_rows = _greedy(n_rows, k)
    best_scope = _scope(best_rows)
    
    # Try to improve with local search / backtracking
    # Use a time-bounded search
    rows = _improve(n_rows, k, best_rows, best_scope, deadline)
    
    return rows


def _scope(rows):
    return max(max(row) for row in rows)


def _validate(rows, n_rows, k):
    """Check that all positive differences are distinct across all rows."""
    diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return len(diffs) == n_rows * k * (k + 1) // 2


def _greedy(n_rows, k):
    """Original greedy: each row takes smallest next mark with unused differences."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _improve(n_rows, k, initial_rows, initial_scope, deadline):
    """
    Try to improve the scope using randomized local search.
    We try to reduce the maximum element by reassigning differences.
    """
    best_rows = list(initial_rows)
    best_scope = initial_scope
    
    # Strategy: try to construct with a tighter scope
    # Work backwards from a candidate scope
    candidate_scope = best_scope - 1
    
    while candidate_scope >= max(1, best_scope - 50) and time.time() < deadline - 5:
        # Try to find a valid configuration with scope <= candidate_scope
        rows = _backtrack_search(n_rows, k, candidate_scope, time.time() < deadline - 10)
        if rows is not None:
            if _validate(rows, n_rows, k):
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_rows = rows
                    best_scope = new_scope
                    candidate_scope = best_scope - 1
                    continue
        candidate_scope -= 1
    
    # Also try randomized restarts of greedy with perturbation
    while time.time() < deadline - 5:
        rows = _greedy_randomized(n_rows, k, best_scope)
        if rows is not None and _validate(rows, n_rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_rows = rows
                best_scope = s
    
    return best_rows


def _greedy_randomized(n_rows, k, max_scope_hint=None):
    """Greedy with random tie-breaking to explore different solutions."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                # Small random perturbation: sometimes skip a valid choice
                if random.random() < 0.05:
                    # Try next few options
                    found = False
                    for try_x in range(x, x + 3):
                        if try_x - row[-1] <= 0:
                            continue
                        new_try = {try_x - a for a in row}
                        if not (new_try & used):
                            row.append(try_x)
                            used |= new_try
                            x = try_x
                            found = True
                            break
                    if found:
                        continue
                row.append(x)
                used |= new
        if max_scope_hint is not None and max(row) > max_scope_hint:
            return None
    return rows


def _backtrack_search(n_rows, k, max_scope, time_limit_check):
    """
    Backtracking search to find rows with max entry <= max_scope.
    We assign differences in order and build rows.
    """
    if not time_limit_check():
        return None
    
    total_diffs = n_rows * k * (k + 1) // 2
    if max_scope * (max_scope + 1) // 2 < total_diffs:
        return None  # Infeasible by counting bound
    
    used = set()
    rows = []
    
    def backtrack(row_idx, row, x):
        if not time_limit_check():
            return None
        if len(row) == k + 1:
            if row_idx == n_rows - 1:
                return rows + [row]
            return backtrack(row_idx + 1, [0], 0)
        
        # Try next positions
        for next_x in range(x + 1, max_scope + 1):
            new = {next_x - a for a in row}
            if new & used:
                continue
            # Pruning: check if remaining differences can fit
            remaining_in_row = k + 1 - len(row)
            if remaining_in_row > 0:
                # The smallest possible max difference from next_x to fill remaining slots
                # is at least remaining_in_row (if we add 1,2,...)
                # This is a weak bound; skip for now
                pass
            
            new_used = used | new
            result = backtrack(row_idx, row + [next_x], next_x)
            if result is not None:
                return result
            used = new_used  # restore
            # Actually we need to be more careful with used
            # Let me restructure
    
    # The above backtracking is complex with used state. Let me use a simpler approach.
    # Instead, let's do a direct backtracking with proper state management.
    
    used = set()
    rows = []
    
    def bt(row_idx, row, current_max):
        if not time_limit_check():
            return False
        if len(row) == k + 1:
            if row_idx == n_rows - 1:
                rows.append(list(row))
                return True
            return bt(row_idx + 1, [0], 0)
        
        start = (row[-1] + 1) if row else 1
        for next_x in range(start, max_scope + 1):
            new = {next_x - a for a in row}
            if new & used:
                continue
            # Check feasibility: can we still place remaining elements?
            rem = k + 1 - len(row)
            if rem > 0:
                # Minimum span needed: at least rem (consecutive integers)
                if next_x + rem - 1 > max_scope:
                    break  # No point trying larger next_x
            used.update(new)
            row.append(next_x)
            if bt(row_idx, row, next_x):
                return True
            row.pop()
            used.difference_update(new)
        
        return False
    
    if bt(0, [0], 0):
        return rows
    return None
# EVOLVE-BLOCK-END
