"""
Maximum subset of {1..192} with no element dividing the sum of any 3 distinct other elements.
Uses ILP formulation with scipy.optimize.milp, plus greedy/local-search fallback.
"""
import time
import sys
import random

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0

N = 192
MAX_K = 48


def _precompute_conflicts():
    """
    Precompute conflict triples.
    For each d in 1..N, find all triples (a,b,c) of distinct values in 1..N
    such that d divides (a+b+c) and d not in {a,b,c}.
    Return: conflicts[d] = list of (a_idx, b_idx, c_idx) where indices are 0-based
    (value = idx+1)
    """
    conflicts = [[] for _ in range(N + 1)]
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            if a == d:
                continue
            for b in range(a + 1, N + 1):
                if b == d:
                    continue
                for c in range(b + 1, N + 1):
                    if c == d:
                        continue
                    if (a + b + c) % d == 0:
                        conflicts[d].append((a - 1, b - 1, c - 1))
    return conflicts


def _greedy_construct(order):
    """Greedy construction with given order."""
    selected = []
    selected_set = set()
    for c in order:
        # Check if c can be added
        valid = True
        n = len(selected)
        if n >= 3:
            # Check 1: c must not divide sum of any 3 in selected
            res_count = [0] * c
            for x in selected:
                res_count[x % c] += 1
            for r1 in range(c):
                if res_count[r1] == 0:
                    continue
                for r2 in range(r1, c):
                    if res_count[r2] == 0:
                        continue
                    r3 = (-r1 - r2) % c
                    if r3 < r2:
                        continue
                    if r1 == r2 == r3:
                        if res_count[r1] >= 3:
                            valid = False
                            break
                    elif r1 == r2:
                        if res_count[r1] >= 2 and res_count[r3] >= 1:
                            valid = False
                            break
                    elif r2 == r3:
                        if res_count[r1] >= 1 and res_count[r2] >= 2:
                            valid = False
                            break
                    else:
                        if res_count[r1] >= 1 and res_count[r2] >= 1 and res_count[r3] >= 1:
                            valid = False
                            break
                if not valid:
                    break
        if not valid:
            continue
        
        # Check 2: For each a in selected, a must not divide (c+x+y) for distinct x,y in selected\{a}
        for idx_a, a in enumerate(selected):
            target = (-c) % a
            res_seen = set()
            for i in range(n):
                if i == idx_a:
                    continue
                x = selected[i]
                xr = x % a
                needed = (target - xr) % a
                if needed in res_seen:
                    valid = False
                    break
                res_seen.add(xr)
            if not valid:
                break
        
        if valid:
            selected.append(c)
            selected_set.add(c)
    return selected


def _validate_set(s):
    """Validate that no element divides the sum of any 3 distinct others."""
    n = len(s)
    for i in range(n):
        d = s[i]
        others = [s[j] for j in range(n) if j != i]
        m = len(others)
        for x in range(m):
            for y in range(x + 1, m):
                for z in range(y + 1, m):
                    if (others[x] + others[y] + others[z]) % d == 0:
                        return False
    return True


def _solve_ilp(conflicts, deadline):
    """
    Solve the ILP using scipy.optimize.milp.
    Variables: x[0..N-1] binary, x[i] = 1 if value (i+1) is selected.
    Objective: maximize sum(x)
    Constraints: for each conflict (d, a, b, c): x[d-1] + x[a-1] + x[b-1] + x[c-1] <= 3
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return []
    
    n_vars = N
    n_conflicts = sum(len(c) for c in conflicts)
    
    # Build constraint matrix
    # Each constraint: x[d-1] + x[a-1] + x[b-1] + x[c-1] <= 3
    # In linear constraint form: A @ x <= b_ub
    rows = []
    for d in range(1, N + 1):
        for (a_idx, b_idx, c_idx) in conflicts[d]:
            row = [0.0] * n_vars
            row[d - 1] = 1.0
            row[a_idx] = 1.0
            row[b_idx] = 1.0
            row[c_idx] = 1.0
            rows.append(row)
    
    if not rows:
        return list(range(1, N + 1))
    
    A_ub = np.array(rows, dtype=float)
    b_ub = np.full(len(rows), 3.0)
    
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(n_vars)
    
    # Bounds: 0 <= x[i] <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality: all binary
    integrality = np.ones(n_vars)
    
    cons = LinearConstraint(A_ub, -np.inf, b_ub)
    bounds = Bounds(lb, ub)
    
    result = milp(
        c=c,
        constraints=cons,
        integrality=integrality,
        bounds=bounds,
    )
    
    if not result.success:
        return []
    
    selected = []
    for i in range(n_vars):
        if result.x[i] > 0.5:
            selected.append(i + 1)
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    best_set = []
    
    # Try ILP
    if time.time() < deadline - 10:
        ilp_result = _solve_ilp(conflicts, deadline)
        if ilp_result and len(ilp_result) > len(best_set):
            best_set = ilp_result
    
    # Greedy multi-start
    random.seed(42)
    
    # Descending
    s1 = _greedy_construct(list(range(N, 0, -1)))
    if len(s1) > len(best_set):
        best_set = s1
    
    # Ascending
    s2 = _greedy_construct(list(range(1, N + 1)))
    if len(s2) > len(best_set):
        best_set = s2
    
    # Random orders
    for i in range(50):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = _greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try to add elements
    if time.time() < deadline - 5:
        best_set_set = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for c in range(N, 0, -1):
                if c in best_set_set:
                    continue
                test_set = best_set + [c]
                if _validate_set(test_set):
                    best_set = test_set
                    best_set_set.add(c)
                    improved = True
                    break
    
    # Final validation
    if best_set and not _validate_set(best_set):
        # Fallback
        best_set = [192, 191, 190]
    
    return best_set
