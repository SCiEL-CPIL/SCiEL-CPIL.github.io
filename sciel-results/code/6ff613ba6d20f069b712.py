"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # Parse inputs
    # distribution_spec: dict with probabilities for each value pair
    # support: list of allowed reports (value pairs) - each report is (v1, v2)
    # constraints: dict with feasibility parameters
    
    # Values are in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    n_vals = len(values)
    
    # Get the support (all possible reports)
    # Each report is a pair (v1, v2) where v1, v2 in {0,1,2,3}
    if support is None or len(support) == 0:
        # Default: all 16 possible reports
        support = [(v1, v2) for v1 in values for v2 in values]
    
    # Ensure support has exactly 16 entries (all value pairs)
    if len(support) != 16:
        # Create standard 16-report support
        support = [(v1, v2) for v1 in values for v2 in values]
    
    # Map reports to indices
    report_to_idx = {r: i for i, r in enumerate(support)}
    n_reports = len(support)  # should be 16
    
    # Get probabilities from distribution_spec
    # distribution_spec could be:
    # - dict mapping (v1, v2) -> probability
    # - dict with 'probabilities' key
    # - list of probabilities
    # - other formats
    
    probs = np.zeros(n_reports)
    
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            prob_data = distribution_spec['probabilities']
            if isinstance(prob_data, dict):
                for (v1, v2), p in prob_data.items():
                    if (v1, v2) in report_to_idx:
                        probs[report_to_idx[(v1, v2)]] = p
            elif isinstance(prob_data, (list, tuple, np.ndarray)):
                for i, p in enumerate(prob_data):
                    if i < n_reports:
                        probs[i] = p
        else:
            # Try to interpret as direct mapping
            for key, p in distribution_spec.items():
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    r = tuple(key)
                    if r in report_to_idx:
                        probs[report_to_idx[r]] = p
                elif isinstance(key, int) and 0 <= key < n_reports:
                    probs[key] = p
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_reports:
            probs = np.array(distribution_spec, dtype=float)
        elif len(distribution_spec) == n_vals * n_vals:
            probs = np.array(distribution_spec, dtype=float).reshape(-1)
    
    # Normalize probabilities if they don't sum to 1
    prob_sum = probs.sum()
    if prob_sum > 1e-12:
        probs = probs / prob_sum
    else:
        # Uniform distribution as fallback
        probs = np.ones(n_reports) / n_reports
    
    # Ensure non-negative
    probs = np.maximum(probs, 0)
    
    # Now solve the LP for optimal mechanism
    # Variables:
    # x[i, j] = probability of allocating good j to bidder when report is i
    # x[i, 0] = allocation of good 1
    # x[i, 1] = allocation of good 2
    # p[i] = payment when report is i
    #
    # Total variables: 16*2 + 16 = 48
    #
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    #
    # Constraints:
    # 1. Feasibility: 0 <= x[i, j] <= 1 for all i, j
    # 2. Individual rationality: v_i * x[i] - p[i] >= 0 for all i
    #    i.e., p[i] <= v1_i * x[i,0] + v2_i * x[i,1]
    # 3. Incentive compatibility (truthfulness):
    #    For each type i and each possible report k:
    #    v_i * x[i] - p[i] >= v_i * x[k] - p[k]
    #    i.e., p[k] - p[i] >= v_i * (x[k] - x[i])
    #    This must hold for all i, k
    
    # Build LP: maximize sum_i probs[i] * p[i]
    # linprog minimizes, so we negate: minimize -sum_i probs[i] * p[i]
    
    # Variable ordering:
    # x[0,0], x[0,1], x[1,0], x[1,1], ..., x[15,0], x[15,1], p[0], p[1], ..., p[15]
    # First 32 vars: allocations (16 reports * 2 goods)
    # Last 16 vars: payments
    
    n_alloc_vars = n_reports * 2  # 32
    n_pay_vars = n_reports  # 16
    n_vars = n_alloc_vars + n_pay_vars  # 48
    
    # Objective: minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[n_alloc_vars + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # We'll build these constraint by constraint
    
    A_ub_rows = []
    b_ub_vals = []
    
    # 1. Feasibility: x[i,j] >= 0 (handled by bounds)
    # 2. Feasibility: x[i,j] <= 1 (handled by bounds)
    
    # 3. Individual rationality: p[i] <= v1_i * x[i,0] + v2_i * x[i,1]
    #    p[i] - v1_i * x[i,0] - v2_i * x[i,1] <= 0
    for i in range(n_reports):
        v1, v2 = support[i]
        row = np.zeros(n_vars)
        row[n_alloc_vars + i] = 1  # p[i]
        row[i * 2] = -v1  # -v1_i * x[i,0]
        row[i * 2 + 1] = -v2  # -v2_i * x[i,1]
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # 4. Incentive compatibility:
    #    For all i (true type) and k (reported type):
    #    p[k] - p[i] - v_i * (x[k] - x[i]) <= 0
    #    i.e., p[k] - p[i] - v1_i*(x[k,0] - x[i,0]) - v2_i*(x[k,1] - x[i,1]) <= 0
    #    = p[k] - p[i] - v1_i*x[k,0] + v1_i*x[i,0] - v2_i*x[k,1] + v2_i*x[i,1] <= 0
    for i in range(n_reports):
        v1_i, v2_i = support[i]
        for k in range(n_reports):
            if i == k:
                continue  # trivially satisfied
            row = np.zeros(n_vars)
            # p[k]
            row[n_alloc_vars + k] = 1
            # -p[i]
            row[n_alloc_vars + i] = -1
            # -v1_i * x[k,0]
            row[k * 2] = -v1_i
            # +v1_i * x[i,0]
            row[i * 2] = v1_i
            # -v2_i * x[k,1]
            row[k * 2 + 1] = -v2_i
            # +v2_i * x[i,1]
            row[i * 2 + 1] = v2_i
            A_ub_rows.append(row)
            b_ub_vals.append(0)
    
    # Bounds for variables
    # Allocations: 0 <= x[i,j] <= 1
    # Payments: 0 <= p[i] <= inf (but IR will bound them)
    bounds = []
    for i in range(n_reports):
        for j in range(2):
            bounds.append((0, 1))
    for i in range(n_reports):
        bounds.append((0, None))
    
    # Solve LP
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    
    if time.time() > deadline:
        # Time's up, return a trivial feasible solution
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return a trivial feasible solution
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x = result.x
    allocations = x[:n_alloc_vars].reshape(n_reports, 2)
    payments = x[n_alloc_vars:n_alloc_vars + n_pay_vars]
    
    # Ensure numerical precision: clip small negative values
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify and clean up
    # Ensure payments are non-negative
    payments = np.maximum(payments, 0)
    
    return {'allocations': allocations, 'payments': payments}
