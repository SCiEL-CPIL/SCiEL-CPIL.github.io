"""
Difference triangle set (7,5) solver.

Strategy:
1. Use scipy MILP to jointly optimize all rows as a binary integer program.
2. Formulate: minimize max entry s.t. all pairwise differences are distinct.
3. Fall back to greedy + local search if MILP fails or times out.
4. Polish with local search on the MILP solution.
"""
import time
import random
import sys
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _solve_milp(n_rows, k, deadline):
    """
    Solve the difference triangle problem using scipy MILP.
    
    Variables:
    - x[r][j] = value at position j in row r (integer, 0 <= x <= max_scope)
    - s = scope (max entry)
    - d[r][i][j] = 1 if difference (x[r][j] - x[r][i]) is used (binary)
    
    Constraints:
    - x[r][0] = 0 for all r
    - x[r][j] > x[r][j-1] for all r, j >= 1
    - x[r][j] <= s for all r, j
    - For each pair (r,i,j) with i<j: d[r][i][j] = 1 iff x[r][j] - x[r][i] is a valid diff
    - No two rows use the same difference: for each diff value v, sum of d[r][i][j] over all (r,i,j) with x[r][j]-x[r][i]=v <= 1
    
    This is complex, so we use a simpler formulation:
    - Minimize s
    - x[r][j] are integers with 0 = x[r][0] < x[r][1] < ... < x[r][k] <= s
    - All pairwise differences within each row are distinct across all rows
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return None
    
    # For (7,5): 7 rows, 6 entries each (positions 0..5)
    # Position 0 is always 0
    # We need to assign positions 1..5 for each row
    
    # Max scope estimate: start with a reasonable upper bound
    # Counting bound is 105, best known is 112
    max_scope = 130
    
    # Number of variables:
    # - s: 1 variable (scope)
    # - x[r][j] for r in 0..n_rows-1, j in 1..k: n_rows * k variables
    n_x = n_rows * k  # x[r][j] for j=1..k
    n_vars = 1 + n_x  # s + x vars
    
    # Objective: minimize s
    c = np.zeros(n_vars)
    c[0] = 1.0  # minimize s
    
    # Bounds:
    # s: 1..max_scope
    # x[r][j]: 1..max_scope (but with ordering constraints)
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, max_scope)
    lb[0] = 1  # s >= 1
    for r in range(n_rows):
        for j in range(1, k + 1):
            idx = 1 + r * k + (j - 1)
            lb[idx] = j  # x[r][j] >= j (since strictly increasing from 0)
    
    # Constraints:
    # 1. Ordering: x[r][j] > x[r][j-1] for j=2..k
    #    x[r][j] - x[r][j-1] >= 1
    A_rows = []
    b_lower = []
    b_upper = []
    
    for r in range(n_rows):
        for j in range(2, k + 1):
            # x[r][j] - x[r][j-1] >= 1
            row = np.zeros(n_vars)
            idx_j = 1 + r * k + (j - 1)
            idx_jm1 = 1 + r * k + (j - 2)
            row[idx_j] = 1.0
            row[idx_jm1] = -1.0
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(np.inf)
    
    # 2. x[r][j] <= s for all r, j
    for r in range(n_rows):
        for j in range(1, k + 1):
            # x[r][j] - s <= 0
            row = np.zeros(n_vars)
            idx_j = 1 + r * k + (j - 1)
            row[idx_j] = 1.0
            row[0] = -1.0
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(0.0)
    
    if A_rows:
        A = np.array(A_rows)
        b_lo = np.array(b_lower)
        b_hi = np.array(b_upper)
    else:
        A = np.empty((0, n_vars))
        b_lo = np.empty(0)
        b_hi = np.empty(0)
    
    # Integrities: all variables are integer
    integrality = np.ones(n_vars)
    
    # Solve
    res = milp(
        c=c,
        constraints=LinearConstraint(A, b_lo, b_hi),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={'time_limit': int(deadline - time.time() - 5), 'disp': False}
    )
    
    if not res.success and res.status != 2:  # status 2 = optimal
        # Check if we got a feasible solution even if not proven optimal
        if res.x is None or not np.isfinite(res.x).all():
            return None
    
    if res.x is None:
        return None
    
    # Extract solution
    s_val = int(round(res.x[0]))
    rows = []
    for r in range(n_rows):
        row = [0]
        for j in range(1, k + 1):
            idx = 1 + r * k + (j - 1)
            row.append(int(round(res.x[idx])))
        rows.append(row)
    
    # Validate
    used_diffs = set()
    for row in rows:
        diffs = _diffs_of_row(row)
        if diffs & used_diffs:
            return None  # Invalid: duplicate differences
        used_diffs |= diffs
    
    scope = _total_scope(rows)
    if scope > s_val + 1:  # Allow small rounding
        return None
    
    return rows


def _generate_row_greedy_pure(used, k):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, frozenset(new_diffs)


def _generate_row_greedy(used, k, rng, pool_size=3):
    """
    Greedily construct a row of k+1 integers starting at 0.
    At each step, pick from the top 'pool_size' smallest valid candidates.
    Returns (row, diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.5:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, frozenset(new_diffs)


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        row_indices = list(range(len(rows)))
        rng.shuffle(row_indices)
        
        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]
            
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break
    
    return rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break
    
    return rows


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    return rows


def _joint_simulated_annealing(n_rows, k, rng, deadline):
    """
    Joint simulated annealing: optimize all rows simultaneously.
    Moves: change a single element in a single row, or swap two elements between rows.
    """
    # Start from greedy solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    temperature = 15.0
    cooling_rate = 0.998
    
    while time.time() < deadline and temperature > 0.05:
        # Pick a random row and position to modify
        idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        row = rows[idx]
        old_val = row[pos]
        
        # Try a random new value (biased toward reduction)
        if rng.random() < 0.7:
            # Try to reduce
            delta = -rng.randint(1, 8)
        else:
            delta = rng.randint(-8, 8)
        new_val = old_val + delta
        
        # Check bounds
        if new_val <= row[pos - 1]:
            continue
        if pos + 1 <= k and new_val >= row[pos + 1]:
            continue
        
        # Compute diff changes
        old_diffs = set()
        for i in range(k + 1):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(k + 1):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)
        
        remaining = used - old_diffs
        if new_diffs & remaining:
            continue
        
        # Compute new scope
        new_scope = max(current_scope, new_val)
        if new_val == current_scope:
            other_max = max(max(r) for i, r in enumerate(rows) if i != idx)
            new_scope = max(other_max, new_val)
        
        # Accept if better or with probability exp(-delta/temperature)
        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        
        temperature *= cooling_rate
    
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    # Step 1: Try MILP solver
    if time.time() < deadline - 60:
        milp_rows = _solve_milp(n_rows, k, deadline - 60)
        if milp_rows is not None:
            best_rows = milp_rows
            best_scope = _total_scope(milp_rows)
    
    # Step 2: Joint simulated annealing
    if time.time() < deadline - 20:
        sa_rows = _joint_simulated_annealing(n_rows, k, rng, deadline - 20)
        if sa_rows is not None:
            sa_scope = _total_scope(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
    
    # Step 3: Polish the best solution found
    if best_rows is not None and time.time() < deadline - 5:
        best_rows = _polish_pass(best_rows, k, deadline - 5)
        best_scope = _total_scope(best_rows)
    
    # Step 4: Fallback to greedy if nothing else worked
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k)
        if best_rows is None:
            # Last resort: brute force greedy
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                best_rows.append(row)
    
    return best_rows

# EVOLVE-BLOCK-END
