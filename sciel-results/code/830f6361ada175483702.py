"""
Incentive-compatible finite auction design.

Strategy: For a single additive bidder with two goods, incentive compatibility
decomposes. We solve a finite LP over allocation probabilities and payments
for all 16 types, maximizing expected revenue subject to IC, IR, and
nonnegativity constraints.
"""
import time
import numpy as np
from scipy.optimize import linprog


def _parse_distribution(distribution_spec, support):
    """
    Parse the distribution specification into a probability vector over the support.
    
    distribution_spec: could be a dict mapping types to probs, a list of probs,
                       or some other structure.
    support: list of types, e.g., [(0,0), (0,1), ..., (3,3)]
    """
    n = len(support)
    
    # Try to extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n)
        for i, t in enumerate(support):
            # Type might be tuple or list
            key = tuple(t) if isinstance(t, (list, tuple)) else t
            if key in distribution_spec:
                probs[i] = distribution_spec[key]
            elif t in distribution_spec:
                probs[i] = distribution_spec[t]
            else:
                # Try string key
                probs[i] = distribution_spec.get(str(t), 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(n) / n
        return probs
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n:
            total = arr.sum()
            if total > 0:
                arr = arr / total
            else:
                arr = np.ones(n) / n
            return arr
        else:
            # Maybe it's a 2D array of probabilities over values
            # e.g., shape (4,4) for values {0,1,2,3} x {0,1,2,3}
            if arr.size == n:
                arr = arr.reshape(n)
                total = arr.sum()
                if total > 0:
                    arr = arr / total
                else:
                    arr = np.ones(n) / n
                return arr
    # Default: uniform
    return np.ones(n) / n


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Build type list
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    # Use support if provided, otherwise use our types
    if support is not None and len(support) > 0:
        # Map support to our type indexing
        type_list = list(support)
        # Create mapping from type to index
        type_to_idx = {}
        for i, t in enumerate(type_list):
            key = tuple(t) if isinstance(t, (list, tuple)) else t
            type_to_idx[key] = i
        n_types = len(type_list)
    else:
        type_list = types
        type_to_idx = {t: i for i, t in enumerate(types)}
        n_types = len(type_list)
    
    # Parse probabilities
    probs = _parse_distribution(distribution_spec, type_list)
    probs = np.asarray(probs, dtype=float).flatten()
    if probs.shape[0] != n_types:
        probs = np.ones(n_types) / n_types
    
    # Ensure probs are valid
    probs = np.clip(probs, 0, 1)
    total = probs.sum()
    if total > 1e-15:
        probs /= total
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables:
    # x[i, g] = allocation probability for good g to type i, for i in range(n_types), g in range(2)
    # p[i] = payment for type i
    # Total variables: n_types * 2 + n_types = 3 * n_types = 48
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # So minimize -sum_i probs[i] * p[i]
    
    # Constraints:
    # 1. Allocation feasibility: 0 <= x[i,g] <= 1 for all i,g
    # 2. Nonnegative payments: p[i] >= 0
    # 3. Individual rationality: For each type i, payment <= expected value
    #    p[i] <= sum_g x[i,g] * v_i[g]
    #    => sum_g x[i,g]*v_i[g] - p[i] >= 0
    # 4. Incentive compatibility: For each pair (i, j), type i reporting as j is not better
    #    u(i, i) >= u(i, j)
    #    u(i, i) = sum_g x[i,g]*v_i[g] - p[i]
    #    u(i, j) = sum_g x[j,g]*v_i[g] - p[j]
    #    u(i, i) - u(i, j) >= 0
    #    sum_g (x[i,g] - x[j,g]) * v_i[g] - p[i] + p[j] >= 0
    #    => -sum_g (x[i,g] - x[j,g]) * v_i[g] + p[i] - p[j] <= 0
    #    Or equivalently: sum_g (x[j,g] - x[i,g]) * v_i[g] + p[i] - p[j] <= 0
    
    # Let's set up the LP:
    # Variables indexed as:
    #   0 .. n_types-1: x[i,0] for i in 0..n_types-1
    #   n_types .. 2*n_types-1: x[i,1] for i in 0..n_types-1
    #   2*n_types .. 3*n_types-1: p[i] for i in 0..n_types-1
    
    def x_idx(i, g):
        return i + g * n_types
    
    def p_idx(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    # Objective: minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: for each pair (i, j), i != j
    # sum_g (x[j,g] - x[i,g]) * v_i[g] + p[i] - p[j] <= 0
    for i in range(n_types):
        v_i = np.array([type_list[i][0], type_list[i][1]], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            for g in range(n_goods):
                row[x_idx(j, g)] += v_i[g]
                row[x_idx(i, g)] -= v_i[g]
            row[p_idx(i)] += 1.0
            row[p_idx(j)] -= 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints: for each type i
    # sum_g x[i,g]*v_i[g] - p[i] >= 0
    # => -sum_g x[i,g]*v_i[g] + p[i] <= 0
    for i in range(n_types):
        v_i = np.array([type_list[i][0], type_list[i][1]], dtype=float)
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[x_idx(i, g)] -= v_i[g]
        row[p_idx(i)] += 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Upper bounds: x[i,g] <= 1
    # This is handled by bounds in linprog
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    
    # Bounds: 0 <= x[i,g] <= 1, p[i] >= 0
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return zero allocations and zero payments
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x
        
        # Extract allocations and payments
        allocations = np.zeros((n_types, n_goods))
        for i in range(n_types):
            for g in range(n_goods):
                allocations[i, g] = x[x_idx(i, g)]
        
        payments = np.zeros(n_types)
        for i in range(n_types):
            payments[i] = x[p_idx(i)]
        
        # Clean up small numerical errors
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.clip(payments, 0.0, None)
        
        # Verify IC and IR with tolerance
        # Check IR: p[i] <= sum_g x[i,g] * v_i[g]
        for i in range(n_types):
            v_i = np.array([type_list[i][0], type_list[i][1]], dtype=float)
            ev = np.dot(allocations[i], v_i)
            if payments[i] > ev + 1e-6:
                # Slight violation, clip payment
                payments[i] = max(0.0, ev)
        
        # Check IC: u(i,i) >= u(i,j) for all i,j
        # Recompute utilities
        utils = np.zeros((n_types, n_types))
        for i in range(n_types):
            v_i = np.array([type_list[i][0], type_list[i][1]], dtype=float)
            for j in range(n_types):
                utils[i, j] = np.dot(allocations[j], v_i) - payments[j]
        
        # Check for violations
        max_violation = 0.0
        for i in range(n_types):
            for j in range(n_types):
                if utils[i, j] > utils[i, i] + 1e-6:
                    max_violation = max(max_violation, utils[i, j] - utils[i, i])
        
        if max_violation > 1e-6:
            # IC violated, try to fix by reducing allocations or increasing payments
            # Simple fix: set all allocations to 0 and payments to 0
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception as e:
        # Fallback on any error
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
