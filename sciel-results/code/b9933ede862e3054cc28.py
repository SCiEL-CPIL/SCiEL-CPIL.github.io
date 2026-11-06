"""Incentive-compatible finite auction design with LP-based optimal mechanism."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    # The report space: (v1, v2) where v1, v2 in {0,1,2,3}
    # Index: i = v1 * 4 + v2
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    n_reports = 16
    
    # Parse distribution_spec into a probability vector over 16 reports
    # distribution_spec could be:
    # - a list/array of 16 probabilities
    # - a dict mapping (v1,v2) -> prob
    # - a list of (v1,v2,prob) tuples
    # - a function
    probs = _parse_distribution(distribution_spec, n_reports)
    
    # Parse support: list of 16 tuples (v1, v2) or similar
    # We assume the standard 16-report support
    if support is not None and isinstance(support, (list, tuple)) and len(support) > 0:
        # Use the provided support ordering
        pass  # We'll use standard ordering for now
    
    # Parse constraints
    # constraints may contain:
    # - 'regret_bound': float (default 1e-8)
    # - 'ir_required': bool (default True)
    # - other constraint specifications
    regret_bound = 1e-8
    ir_required = True
    if constraints is not None and isinstance(constraints, dict):
        regret_bound = constraints.get('regret_bound', 1e-8)
        ir_required = constraints.get('ir_required', True)
    
    # Solve the LP for optimal expected revenue
    # Variables: x_{i,j} = allocation probability of good j to bidder when report is i
    # For each report i = (v1_i, v2_i), we have 2 allocation variables (one per good)
    # Total allocation variables: 16 * 2 = 32
    # Payment variables: 16 (one per report)
    # Total variables: 48
    
    # Objective: maximize expected revenue = sum_i p_i * payment_i
    # In LP form (minimize): -sum_i p_i * payment_i
    
    # IC constraints: For each pair of reports (i, j), the expected utility of reporting i
    # when true type is j must be >= expected utility of reporting j when true type is j
    # U(i, j) = v1_j * x_{i,0} + v2_j * x_{i,1} - payment_i
    # IC: U(j, j) >= U(i, j) for all i, j
    # => v1_j * x_{j,0} + v2_j * x_{j,1} - payment_j >= v1_j * x_{i,0} + v2_j * x_{i,1} - payment_i
    
    # IR constraints: U(j, j) >= 0 for all j
    # => v1_j * x_{j,0} + v2_j * x_{j,1} - payment_j >= 0
    
    # Feasibility: 0 <= x_{i,j} <= 1 for all i, j
    # Nonnegative payments: payment_i >= 0
    
    # Let's set up the LP
    # x variables: x[0..31] where x[i*2 + g] = allocation of good g at report i
    # p variables: p[0..15] where p[i] = payment at report i
    # Total: 48 variables
    
    n_alloc = n_reports * 2  # 32
    n_pay = n_reports  # 16
    n_vars = n_alloc + n_pay  # 48
    
    # Objective: minimize -sum_i probs[i] * payment[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[n_alloc + i] = -probs[i]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    # IC constraints: For each pair (i, j):
    # U(j,j) - U(i,j) >= 0
    # (v1_j * x_{j,0} + v2_j * x_{j,1} - payment_j) - (v1_j * x_{i,0} + v2_j * x_{i,1} - payment_i) >= 0
    # => v1_j * (x_{j,0} - x_{i,0}) + v2_j * (x_{j,1} - x_{i,1}) - payment_j + payment_i >= 0
    # In A_ub form: -[v1_j * (x_{j,0} - x_{i,0}) + v2_j * (x_{j,1} - x_{i,1}) - payment_j + payment_i] <= 0
    
    A_rows = []
    b_rows = []
    
    for j in range(n_reports):  # true type
        v1_j, v2_j = values[j]
        for i in range(n_reports):  # reported type
            if i == j:
                continue
            # Constraint: U(j,j) >= U(i,j)
            # v1_j * x_{j,0} + v2_j * x_{j,1} - payment_j >= v1_j * x_{i,0} + v2_j * x_{i,1} - payment_i
            # Rearranged: v1_j*(x_{j,0} - x_{i,0}) + v2_j*(x_{j,1} - x_{i,1}) + payment_i - payment_j >= 0
            # A_ub form: -(v1_j*(x_{j,0} - x_{i,0}) + v2_j*(x_{j,1} - x_{i,1}) + payment_i - payment_j) <= 0
            row = np.zeros(n_vars)
            # Coefficient for x_{j,0}: v1_j
            row[j * 2 + 0] = -v1_j
            # Coefficient for x_{j,1}: v2_j
            row[j * 2 + 1] = -v2_j
            # Coefficient for x_{i,0}: -v1_j
            row[i * 2 + 0] = v1_j
            # Coefficient for x_{i,1}: -v2_j
            row[i * 2 + 1] = v2_j
            # Coefficient for payment_i: +1
            row[n_alloc + i] = -1
            # Coefficient for payment_j: -1
            row[n_alloc + j] = 1
            A_rows.append(row)
            b_rows.append(0)
    
    # IR constraints: U(j,j) >= 0
    # v1_j * x_{j,0} + v2_j * x_{j,1} - payment_j >= 0
    # A_ub form: -(v1_j * x_{j,0} + v2_j * x_{j,1} - payment_j) <= 0
    if ir_required:
        for j in range(n_reports):
            v1_j, v2_j = values[j]
            row = np.zeros(n_vars)
            row[j * 2 + 0] = -v1_j
            row[j * 2 + 1] = -v2_j
            row[n_alloc + j] = 1
            A_rows.append(row)
            b_rows.append(0)
    
    # Build A_ub and b_ub
    if A_rows:
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds:
    # Allocation variables: 0 <= x <= 1
    # Payment variables: payment >= 0 (no upper bound)
    bounds = []
    for i in range(n_alloc):
        bounds.append((0, 1))
    for i in range(n_pay):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple mechanism
        return _simple_fallback(probs, values, n_reports)
    
    # Extract solution
    x_sol = result.x[:n_alloc].reshape(n_reports, 2)
    p_sol = result.x[n_alloc:n_alloc + n_pay]
    
    # Clip small negative values due to numerical errors
    x_sol = np.clip(x_sol, 0, 1)
    p_sol = np.clip(p_sol, 0, None)
    
    # Verify constraints
    if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required):
        # Try to fix or use fallback
        x_sol, p_sol = _fix_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required)
        if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required):
            return _simple_fallback(probs, values, n_reports)
    
    return {
        'allocations': x_sol,
        'payments': p_sol
    }


def _parse_distribution(distribution_spec, n_reports):
    """Parse distribution_spec into a probability vector of length n_reports."""
    if distribution_spec is None:
        # Uniform distribution
        return np.ones(n_reports) / n_reports
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_reports,):
            # Normalize
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_reports) / n_reports
        elif len(arr) == n_reports:
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_reports) / n_reports
        elif arr.ndim == 2 and arr.shape[0] == n_reports:
            # Could be (v1, v2, prob) triples
            probs = np.zeros(n_reports)
            for row in arr:
                if len(row) >= 3:
                    v1, v2, p = int(row[0]), int(row[1]), float(row[2])
                    idx = v1 * 4 + v2
                    if 0 <= idx < n_reports:
                        probs[idx] = p
            total = probs.sum()
            if total > 0:
                return probs / total
            return np.ones(n_reports) / n_reports
    
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_reports)
        for key, val in distribution_spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = int(key[0]), int(key[1])
                idx = v1 * 4 + v2
                if 0 <= idx < n_reports:
                    probs[idx] = float(val)
            elif isinstance(key, int) and 0 <= key < n_reports:
                probs[key] = float(val)
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.ones(n_reports) / n_reports
    
    if callable(distribution_spec):
        # Try calling it
        try:
            result = distribution_spec()
            return _parse_distribution(result, n_reports)
        except:
            pass
        # Try with arguments
        try:
            result = distribution_spec(range(4), range(4))
            return _parse_distribution(result, n_reports)
        except:
            pass
    
    # Default: uniform
    return np.ones(n_reports) / n_reports


def _verify_constraints(allocations, payments, probs, values, n_reports, regret_bound, ir_required):
    """Verify IC, IR, feasibility, and nonnegative payments."""
    # Feasibility: 0 <= allocations <= 1
    if np.any(allocations < -1e-9) or np.any(allocations > 1 + 1e-9):
        return False
    
    # Nonnegative payments
    if np.any(payments < -1e-9):
        return False
    
    # IR: U(j,j) >= 0 for all j
    if ir_required:
        for j in range(n_reports):
            v1_j, v2_j = values[j]
            utility = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
            if utility < -1e-9:
                return False
    
    # IC: U(j,j) >= U(i,j) for all i, j
    for j in range(n_reports):
        v1_j, v2_j = values[j]
        u_true = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
        for i in range(n_reports):
            if i == j:
                continue
            u_misrep = v1_j * allocations[i, 0] + v2_j * allocations[i, 1] - payments[i]
            if u_true - u_misrep < -1e-9:
                return False
    
    return True


def _fix_constraints(allocations, payments, probs, values, n_reports, regret_bound, ir_required):
    """Attempt to fix small constraint violations."""
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    return allocations, payments


def _simple_fallback(probs, values, n_reports):
    """Simple fallback: always allocate both goods, charge 0."""
    allocations = np.ones((n_reports, 2))
    payments = np.zeros(n_reports)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
