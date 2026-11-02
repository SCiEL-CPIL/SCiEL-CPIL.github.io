# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    Uses MILP for exact optimization if problem size is small, otherwise greedy.
    """
    # 1. Get a valid initial solution using greedy
    greedy_rows = _greedy_deterministic(n_rows, k)
    if greedy_rows is None:
        return None
        
    T_init = max(max(r) for r in greedy_rows)
    
    # Check if MILP is feasible for this size
    # n_diffs = n_rows * k * (k+1) / 2
    n_diffs = n_rows * k * (k + 1) // 2
    
    # Heuristic: If n_diffs is too large, the B variables explode (N^2)
    # For n_diffs = 105, B vars = ~5000. Total vars ~ 5000.
    # For n_diffs = 200, B vars = ~20000. Total vars ~ 20000.
    # Scipy MILP might struggle with 20k vars + 40k constraints.
    # Let's limit to n_diffs <= 150.
    
    if n_diffs > 150:
        return greedy_rows
    
    try:
        return _solve_milp(n_rows, k, T_init)
    except Exception:
        return greedy_rows

def _solve_milp(n_rows, k, T_init):
    # Variables:
    # x[i, j] for i in range(n_rows), j in range(k+1)
    # T
    # D[i, p, q] for i in range(n_rows), 0 <= p < q <= k
    # B[u, v] for 0 <= u < v < n_diffs_total
    
    n_x = n_rows * (k + 1)
    T_idx = n_x
    
    # List of diff tuples (i, p, q)
    diff_list = []
    for i in range(n_rows):
        for p in range(k + 1):
            for q in range(p + 1, k + 1):
                diff_list.append((i, p, q))
                
    n_diffs = len(diff_list)
    D_start = n_x + 1
    D_idx = {}
    for idx, (i, p, q) in enumerate(diff_list):
        D_idx[(i, p, q)] = D_start + idx
        
    B_start = D_start + n_diffs
    
    # Number of B variables
    n_pairs = n_diffs * (n_diffs - 1) // 2
    B_idx = {}
    pair_count = 0
    for u in range(n_diffs):
        for v in range(u + 1, n_diffs):
            B_idx[(u, v)] = B_start + pair_count
            pair_count += 1
            
    total_vars = B_start + n_pairs
    
    # Objective: Minimize T
    c = np.zeros(total_vars)
    c[T_idx] = 1.0
    
    # Bounds
    # x[i,j] in [0, T_init]
    # T in [0, T_init]
    # D in [1, T_init]
    # B in [0, 1]
    
    lb = np.zeros(total_vars)
    ub = np.zeros(total_vars)
    
    for i in range(n_rows):
        for j in range(k + 1):
            idx = i * (k + 1) + j
            lb[idx] = 0
            ub[idx] = T_init
            
    lb[T_idx] = 0
    ub[T_idx] = T_init
    
    for idx in range(D_start, D_start + n_diffs):
        lb[idx] = 1
        ub[idx] = T_init
        
    for idx in range(B_start, total_vars):
        lb[idx] = 0
        ub[idx] = 1
        
    bounds = Bounds(lb, ub)
    
    # Integrality
    integrality = np.ones(total_vars) # All integer
    
    # Constraints
    A_list = []
    b_lower_list = []
    b_upper_list = []
    
    def add_constraint(row, lower, upper):
        A_list.append(row)
        b_lower_list.append(lower)
        b_upper_list.append(upper)
        
    # 1. x[i, 0] = 0
    for i in range(n_rows):
        row = np.zeros(total_vars)
        idx = i * (k + 1) + 0
        row[idx] = 1.0
        add_constraint(row, 0, 0)
        
    # 2. x[i, j] + 1 <= x[i, j+1]  =>  x[i, j] - x[i, j+1] <= -1
    for i in range(n_rows):
        for j in range(k):
            row = np.zeros(total_vars)
            idx1 = i * (k + 1) + j
            idx2 = i * (k + 1) + j + 1
            row[idx1] = 1.0
            row[idx2] = -1.0
            add_constraint(row, -np.inf, -1)
            
    # 3. x[i, j] <= T  =>  x[i, j] - T <= 0
    for i in range(n_rows):
        for j in range(k + 1):
            row = np.zeros(total_vars)
            idx = i * (k + 1) + j
            row[idx] = 1.0
            row[T_idx] = -1.0
            add_constraint(row, -np.inf, 0)
            
    # 4. D[i, p, q] - x[i, q] + x[i, p] = 0
    for (i, p, q) in diff_list:
        row = np.zeros(total_vars)
        d_idx = D_idx[(i, p, q)]
        x_q_idx = i * (k + 1) + q
        x_p_idx = i * (k + 1) + p
        
        row[d_idx] = 1.0
        row[x_q_idx] = -1.0
        row[x_p_idx] = 1.0
        
        add_constraint(row, 0, 0)
        
    # 5. Distinctness of D
    # For each pair (u, v), D_u != D_v
    # D_u <= D_v - 1 + T_init * (1 - B_uv)
    # D_v <= D_u - 1 + T_init * B_uv
    
    # We need to map diff_list indices to D indices
    # diff_list[u] corresponds to D_start + u
    
    for u in range(n_diffs):
        for v in range(u + 1, n_diffs):
            b_idx = B_idx[(u, v)]
            d_u_idx = D_start + u
            d_v_idx = D_start + v
            
            # Constraint 1: D_u - D_v + T_init * B_uv <= T_init - 1
            # D_u <= D_v - 1 + T_init * B_uv
            # D_u - D_v - T_init * B_uv <= -1
            # Wait:
            # If B=0 (D_u < D_v): D_u <= D_v - 1.
            # If B=1 (D_u > D_v): D_u <= D_v - 1 + T_init. (Always true since D_u <= T_init, D_v >= 1)
            
            row1 = np.zeros(total_vars)
            row1[d_u_idx] = 1.0
            row1[d_v_idx] = -1.0
            row1[b_idx] = -T_init
            add_constraint(row1, -np.inf, -1)
            
            # Constraint 2: D_v <= D_u - 1 + T_init * (1 - B_uv)
            # D_v - D_u + T_init * B_uv <= T_init - 1
            row2 = np.zeros(total_vars)
            row2[d_v_idx] = 1.0
            row2[d_u_idx] = -1.0
            row2[b_idx] = T_init
            add_constraint(row2, -np.inf, T_init - 1)
            
    A = np.array(A_list)
    b_lower = np.array(b_lower_list)
    b_upper = np.array(b_upper_list)
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        raise Exception(f"MILP failed: {res.message}")
        
    x_sol = res.x
    
    # Extract rows
    rows = []
    for i in range(n_rows):
        row = [int(round(x_sol[i * (k + 1) + j])) for j in range(k + 1)]
        rows.append(row)
        
    # Verify
    if not _is_valid(rows, n_rows, k):
        raise Exception("MILP solution invalid")
        
    return rows

def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy_random(n_rows, k, rng, upper_bound):
    # Kept for signature compatibility, but not used in main logic
    pass

def _greedy_deterministic(n_rows, k):
    """Original greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
