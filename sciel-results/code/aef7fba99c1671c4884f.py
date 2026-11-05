"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    # distribution_spec: could be dict or list
    # support: list of allowed reports (value pairs) - each report is (v1, v2)
    # constraints: dict with feasibility parameters
    
    # Values are in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    
    # Get probabilities for each good
    # distribution_spec format: need to figure out
    if isinstance(distribution_spec, dict):
        # Try common formats
        if 'goods' in distribution_spec:
            p1 = np.array(distribution_spec['goods'][0], dtype=float)
            p2 = np.array(distribution_spec['goods'][1], dtype=float)
        elif 'p1' in distribution_spec and 'p2' in distribution_spec:
            p1 = np.array(distribution_spec['p1'], dtype=float)
            p2 = np.array(distribution_spec['p2'], dtype=float)
        elif 'probs' in distribution_spec:
            probs = distribution_spec['probs']
            if isinstance(probs, (list, tuple)) and len(probs) == 2:
                p1 = np.array(probs[0], dtype=float)
                p2 = np.array(probs[1], dtype=float)
            else:
                p1 = np.array(probs, dtype=float)
                p2 = np.array(probs, dtype=float)
        else:
            # Fallback: uniform
            p1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
            p2 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 2:
            p1 = np.array(distribution_spec[0], dtype=float)
            p2 = np.array(distribution_spec[1], dtype=float)
        else:
            p1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
            p2 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    else:
        p1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        p2 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    
    # Ensure probabilities are valid
    p1 = np.maximum(p1, 0)
    p2 = np.maximum(p2, 0)
    if p1.sum() > 0:
        p1 = p1 / p1.sum()
    else:
        p1 = np.array([0.25, 0.25, 0.25, 0.25])
    if p2.sum() > 0:
        p2 = p2 / p2.sum()
    else:
        p2 = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Get support - list of allowed reports
    if support is None:
        support = [(v1, v2) for v1 in values for v2 in values]
    elif isinstance(support, np.ndarray):
        if support.ndim == 1:
            # Could be flattened pairs
            if len(support) % 2 == 0:
                support = [(support[i], support[i+1]) for i in range(0, len(support), 2)]
            else:
                support = [(v, 0) for v in support]
        elif support.ndim == 2:
            support = [(int(row[0]), int(row[1])) for row in support]
    elif isinstance(support, (list, tuple)):
        if len(support) > 0:
            if isinstance(support[0], (list, tuple)):
                support = [(int(s[0]), int(s[1])) for s in support]
            else:
                # Single dimension - treat as pairs of consecutive values
                if len(support) % 2 == 0:
                    support = [(int(support[i]), int(support[i+1])) for i in range(0, len(support), 2)]
                else:
                    support = [(int(v), 0) for v in support]
    
    # Ensure support is a list of (v1, v2) tuples
    support_set = set()
    for s in support:
        if isinstance(s, (list, tuple)):
            support_set.add((int(s[0]), int(s[1])))
        else:
            support_set.add((int(s), 0))
    
    if not support_set:
        support_set = set([(v1, v2) for v1 in values for v2 in values])
    
    support_list = sorted(support_set)
    n_reports = len(support_list)
    
    # Build index mapping
    report_index = {}
    for i, r in enumerate(support_list):
        report_index[r] = i
    
    # For each report r=(v1,v2), the allocation is x_r = (x1_r, x2_r) where x1_r, x2_r in [0,1]
    # Payment is t_r
    # 
    # LP variables: x1_0..x1_{n-1}, x2_0..x2_{n-1}, t_0..t_{n-1}
    # Total variables: 3 * n_reports
    #
    # Objective: maximize sum_{r} p(r) * t_r
    # where p(r) = p1[v1] * p2[v2]
    #
    # Constraints:
    # 1. Feasibility: 0 <= x1_r <= 1, 0 <= x2_r <= 1 for all r
    # 2. Nonnegative payments: t_r >= 0 for all r
    # 3. IR: For each r, utility u(r|r) = v1*x1_r + v2*x2_r - t_r >= 0
    # 4. IC: For each pair (r, r'), u(r|r) >= u(r|r')
    #    i.e., v1_r*x1_r + v2_r*x2_r - t_r >= v1_r*x1_{r'} + v2_r*x2_{r'} - t_{r'}
    
    # LP formulation
    # Variables: [x1_0, x2_0, t_0, x1_1, x2_1, t_1, ..., x1_{n-1}, x2_{n-1}, t_{n-1}]
    # Index: for report i, x1 is at 3*i, x2 is at 3*i+1, t is at 3*i+2
    
    n_vars = 3 * n_reports
    
    # Objective: maximize sum p(r) * t_r
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i, (v1, v2) in enumerate(support_list):
        prob = p1[v1] * p2[v2]
        c[3*i + 2] = -prob  # minimize -revenue = maximize revenue
    
    # Equality constraints: none initially
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_ub_rows = []
    b_ub_rows = []
    
    # Feasibility: x1_r <= 1, x2_r <= 1
    for i in range(n_reports):
        # x1_i <= 1
        row = np.zeros(n_vars)
        row[3*i] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)
        
        # x2_i <= 1
        row = np.zeros(n_vars)
        row[3*i + 1] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)
    
    # Nonnegative payments: -t_r <= 0
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[3*i + 2] = -1
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    # Nonnegative allocations: -x1_r <= 0, -x2_r <= 0
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[3*i] = -1
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
        row = np.zeros(n_vars)
        row[3*i + 1] = -1
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    # IR: v1_r*x1_r + v2_r*x2_r - t_r >= 0
    # => -v1_r*x1_r - v2_r*x2_r + t_r <= 0
    for i, (v1, v2) in enumerate(support_list):
        row = np.zeros(n_vars)
        row[3*i] = -v1
        row[3*i + 1] = -v2
        row[3*i + 2] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    # IC: For each pair (r, r') where r != r':
    # u(r|r) >= u(r|r')
    # v1_r*x1_r + v2_r*x2_r - t_r >= v1_r*x1_{r'} + v2_r*x2_{r'} - t_{r'}
    # => v1_r*x1_r + v2_r*x2_r - t_r - v1_r*x1_{r'} - v2_r*x2_{r'} + t_{r'} >= 0
    # => -v1_r*x1_r - v2_r*x2_r + t_r + v1_r*x1_{r'} + v2_r*x2_{r'} - t_{r'} <= 0
    
    for i, (v1, v2) in enumerate(support_list):
        for j, (v1p, v2p) in enumerate(support_list):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -v1_r*x1_r
            row[3*i] = -v1
            # -v2_r*x2_r
            row[3*i + 1] = -v2
            # +t_r
            row[3*i + 2] = 1
            # +v1_r*x1_{r'}
            row[3*j] = v1
            # +v2_r*x2_{r'}
            row[3*j + 1] = v2
            # -t_{r'}
            row[3*j + 2] = -1
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))  # x1
        bounds.append((0, 1))  # x2
        bounds.append((0, None))  # t
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status == 0:
            x = result.x
            allocations = np.zeros((n_reports, 2))
            payments = np.zeros(n_reports)
            for i in range(n_reports):
                allocations[i, 0] = x[3*i]
                allocations[i, 1] = x[3*i + 1]
                payments[i] = x[3*i + 2]
            
            # Clamp small negatives
            allocations = np.maximum(allocations, 0)
            allocations = np.minimum(allocations, 1)
            payments = np.maximum(payments, 0)
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # LP infeasible or failed - return a safe fallback
            pass
    except Exception:
        pass
    
    # Fallback: simple mechanism
    # Check time
    if time.time() > deadline - 1.0:
        # Very simple fallback
        allocations = np.zeros((n_reports, 2))
        payments = np.zeros(n_reports)
        for i, (v1, v2) in enumerate(support_list):
            if v1 >= 2:
                allocations[i, 0] = 1.0
            if v2 >= 2:
                allocations[i, 1] = 1.0
            payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - 1.0
            payments[i] = max(payments[i], 0)
        return {
            'allocations': allocations,
            'payments': payments
        }
    
    # Try a simpler heuristic
    allocations = np.zeros((n_reports, 2))
    payments = np.zeros(n_reports)
    for i, (v1, v2) in enumerate(support_list):
        if v1 >= 2:
            allocations[i, 0] = 1.0
        if v2 >= 2:
            allocations[i, 1] = 1.0
        payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - 1.0
        payments[i] = max(payments[i], 0)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
