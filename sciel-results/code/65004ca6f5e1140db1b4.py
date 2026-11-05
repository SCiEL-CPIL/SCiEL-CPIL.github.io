"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin
    
    # Values are in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    num_values = len(values)
    
    # --- Parse Distribution Spec ---
    # Goal: Extract p1 (prob for good 1) and p2 (prob for good 2)
    # Expected format: distribution_spec is often a dict or list of lists.
    # Common formats:
    # 1. {'p1': [...], 'p2': [...]}
    # 2. {'goods': [[...], [...]]}
    # 3. [[...], [...]]
    # 4. {'probs': [[...], [...]]}
    
    p1 = None
    p2 = None
    
    if isinstance(distribution_spec, dict):
        if 'p1' in distribution_spec and 'p2' in distribution_spec:
            p1 = np.array(distribution_spec['p1'], dtype=float)
            p2 = np.array(distribution_spec['p2'], dtype=float)
        elif 'goods' in distribution_spec:
            goods = distribution_spec['goods']
            if isinstance(goods, (list, tuple)) and len(goods) >= 2:
                p1 = np.array(goods[0], dtype=float)
                p2 = np.array(goods[1], dtype=float)
        elif 'probs' in distribution_spec:
            probs = distribution_spec['probs']
            if isinstance(probs, (list, tuple)) and len(probs) >= 2:
                p1 = np.array(probs[0], dtype=float)
                p2 = np.array(probs[1], dtype=float)
        elif 'distribution' in distribution_spec:
             # Sometimes nested under 'distribution'
             dist = distribution_spec['distribution']
             if isinstance(dist, dict) and 'p1' in dist and 'p2' in dist:
                 p1 = np.array(dist['p1'], dtype=float)
                 p2 = np.array(dist['p2'], dtype=float)
             elif isinstance(dist, (list, tuple)) and len(dist) >= 2:
                 p1 = np.array(dist[0], dtype=float)
                 p2 = np.array(dist[1], dtype=float)
        
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) >= 2:
            # Assume first two elements are p1 and p2
            p1 = np.array(distribution_spec[0], dtype=float)
            p2 = np.array(distribution_spec[1], dtype=float)
        elif len(distribution_spec) == 1:
            # Single list? Maybe flattened or just one good? 
            # Task says "one additive bidder, two goods", so this is likely malformed or a single good spec.
            # Fallback to uniform if ambiguous.
            p1 = None
            p2 = None

    # Validate and normalize probabilities
    def normalize_prob(p):
        if p is None:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        p = np.array(p, dtype=float).flatten()
        if len(p) != num_values:
            # If length doesn't match, try to map if possible, else uniform
            # Assuming indices 0..3 correspond to values 0..3
            if len(p) > num_values:
                p = p[:num_values]
            else:
                # Pad with zeros
                new_p = np.zeros(num_values)
                new_p[:len(p)] = p
                p = new_p
        p = np.maximum(p, 0)
        if p.sum() > 0:
            p = p / p.sum()
        else:
            p = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        return p

    p1 = normalize_prob(p1)
    p2 = normalize_prob(p2)

    # --- Parse Support ---
    # Goal: List of allowed reports (v1, v2). If not provided, use full support {0,1,2,3}x{0,1,2,3}.
    if support is None:
        support = [(v1, v2) for v1 in values for v2 in values]
    elif isinstance(support, np.ndarray):
        if support.ndim == 2 and support.shape[1] == 2:
            support = [(int(row[0]), int(row[1])) for row in support]
        elif support.ndim == 1:
            # Flattened pairs?
            if len(support) % 2 == 0:
                support = [(int(support[i]), int(support[i+1])) for i in range(0, len(support), 2)]
            else:
                support = [(int(v), 0) for v in support]
        else:
            support = [(v1, v2) for v1 in values for v2 in values]
    elif isinstance(support, (list, tuple)):
        if len(support) > 0:
            if isinstance(support[0], (list, tuple, np.ndarray)):
                support = [(int(s[0]), int(s[1])) for s in support]
            else:
                # Flat list of values? Assume pairs
                if len(support) % 2 == 0:
                    support = [(int(support[i]), int(support[i+1])) for i in range(0, len(support), 2)]
                else:
                    support = [(int(v), 0) for v in support]
        else:
            support = [(v1, v2) for v1 in values for v2 in values]
    
    # Ensure support is a list of unique (v1, v2) tuples
    support_set = set()
    for s in support:
        if isinstance(s, (list, tuple, np.ndarray)):
            if len(s) >= 2:
                support_set.add((int(s[0]), int(s[1])))
        else:
            support_set.add((int(s), 0))
    
    if not support_set:
        support_set = set([(v1, v2) for v1 in values for v2 in values])
        
    # Sort for deterministic indexing
    support_list = sorted(support_set)
    n_reports = len(support_list)
    
    # Build index mapping
    report_index = {}
    for i, r in enumerate(support_list):
        report_index[r] = i

    # --- LP Formulation ---
    # Variables: x1_i, x2_i, t_i for each report i
    # Total variables: 3 * n_reports
    
    n_vars = 3 * n_reports
    
    # Objective: Maximize Expected Revenue = sum p(r) * t_r
    # Minimize -Revenue
    c = np.zeros(n_vars)
    for i, (v1, v2) in enumerate(support_list):
        prob = p1[v1] * p2[v2]
        c[3*i + 2] = -prob
    
    A_ub_rows = []
    b_ub_rows = []
    
    # 1. Feasibility: 0 <= x <= 1 (handled by bounds, but x<=1 is explicit in bounds)
    # Bounds:
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))  # x1_i
        bounds.append((0, 1))  # x2_i
        bounds.append((0, None))  # t_i >= 0
    
    # 2. Individual Rationality (IR):
    # u(r|r) = v1*x1_r + v2*x2_r - t_r >= 0
    # => -v1*x1_r - v2*x2_r + t_r <= 0
    for i, (v1, v2) in enumerate(support_list):
        row = np.zeros(n_vars)
        row[3*i] = -v1
        row[3*i + 1] = -v2
        row[3*i + 2] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
    # 3. Incentive Compatibility (IC):
    # u(r|r) >= u(r|r') for all r, r'
    # v1_r*x1_r + v2_r*x2_r - t_r >= v1_r*x1_{r'} + v2_r*x2_{r'} - t_{r'}
    # => -v1_r*x1_r - v2_r*x2_r + t_r + v1_r*x1_{r'} + v2_r*x2_{r'} - t_{r'} <= 0
    
    # Optimization: Only need to check adjacent or relevant reports? 
    # For finite support with linear utilities, checking all pairs is safest.
    # N=16 is small, so N^2 = 256 constraints is fine.
    
    for i, (v1, v2) in enumerate(support_list):
        for j, (v1p, v2p) in enumerate(support_list):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # LHS: u(r|r) - u(r|r') <= 0
            # Term for r (own report):
            row[3*i] = -v1
            row[3*i + 1] = -v2
            row[3*i + 2] = 1
            # Term for r' (other report):
            row[3*j] = v1
            row[3*j + 1] = v2
            row[3*j + 2] = -1
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Solve LP
    try:
        # Use highs method for speed and robustness
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={"presolve": True})
        
        if result.status == 0:
            x = result.x
            allocations = np.zeros((n_reports, 2))
            payments = np.zeros(n_reports)
            for i in range(n_reports):
                allocations[i, 0] = x[3*i]
                allocations[i, 1] = x[3*i + 1]
                payments[i] = x[3*i + 2]
            
            # Clamp small numerical errors
            allocations = np.maximum(allocations, 0)
            allocations = np.minimum(allocations, 1)
            payments = np.maximum(payments, 0)
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # LP failed or infeasible
            pass
    except Exception as e:
        # Fallback on exception
        pass
    
    # --- Fallback Mechanism ---
    # If LP fails, return a simple, valid mechanism.
    # A simple valid mechanism: Always allocate nothing, pay nothing.
    # Revenue = 0. Valid (IR: 0>=0, IC: 0>=0).
    # Or a slightly better heuristic: Allocate if value > threshold.
    # However, threshold mechanisms are not generally IC without careful payment design.
    # The "null mechanism" is always safe.
    
    if time.time() > deadline - 1.0:
         # Very simple fallback
         allocations = np.zeros((n_reports, 2))
         payments = np.zeros(n_reports)
         return {
             'allocations': allocations,
             'payments': payments
         }
    
    # Try a simple dominant strategy mechanism:
    # Good 1: Allocate 1 if v1 >= 2, pay v1 if allocated? No, that's not IC.
    # Let's stick to the null mechanism for safety if LP fails, as it guarantees validity.
    allocations = np.zeros((n_reports, 2))
    payments = np.zeros(n_reports)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
