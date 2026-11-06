"""Incentive-compatible finite auction design via direct LP solve with tight tolerances."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
        
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=np.float64)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=np.float64)
        else:
            # Try to find a list of probabilities
            probs = None
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 16:
                    probs = np.array(val, dtype=np.float64)
                    break
            if probs is None:
                probs = np.ones(16, dtype=np.float64) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=np.float64)
        if len(probs) != 16:
            probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Support: 16 type reports, each is (v1, v2) where v1, v2 in {0, 1, 2, 3}
    if support is None or len(support) != 16:
        support = [(i // 4, i % 4) for i in range(16)]
    
    # Ensure support is a list of tuples
    support_list = []
    for s in support:
        if isinstance(s, (list, tuple, np.ndarray)):
            support_list.append((float(s[0]), float(s[1])))
        else:
            support_list.append((float(s), 0.0))
    
    # Parse constraints
    max_payment = None
    min_revenue = None
    if isinstance(constraints, dict):
        max_payment = constraints.get('max_payment', None)
        min_revenue = constraints.get('min_revenue', None)
    
    # Build LP:
    # Variables: x[0:16] = allocation of good 1 for each type, x[16:32] = allocation of good 2, x[32:48] = payments
    # Objective: maximize sum_i probs[i] * payment[i]
    # Constraints:
    # 1. Feasibility: 0 <= x[good, type] <= 1
    # 2. Non-negative payments: x[32:] >= 0
    # 3. Individual Rationality: payment[t] <= v_t . allocation[t] for all t
    # 4. Incentive Compatibility (monotonicity + payment formula)
    
    n_types = 16
    n_goods = 2
    
    # For a single bidder with additive values, truthful mechanism characterization:
    # A mechanism is truthful iff:
    # (a) Allocation is monotone in each value coordinate
    # (b) Payment = value . allocation - integral of allocation over value space (for continuous)
    # For finite support, we use the LP formulation directly.
    
    # Direct LP approach:
    # Variables: a[t][g] for t in range(16), g in range(2), and p[t] for t in range(16)
    # Total variables: 16*2 + 16 = 48
    
    n_vars = n_types * n_goods + n_types  # 48 variables
    
    # Objective: maximize sum_t probs[t] * p[t]
    # In linprog, we minimize, so c = -probs[t] for payment variables
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[2 * n_types + t] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    # Constraint 1: Individual Rationality
    # p[t] <= v1[t] * a[t][0] + v2[t] * a[t][1]
    # => p[t] - v1[t]*a[t][0] - v2[t]*a[t][1] <= 0
    
    # Constraint 2: Incentive Compatibility
    # For all type pairs (t, t'), p[t] - v_t . a[t] <= p[t'] - v_t . a[t']
    # This is the key IC constraint.
    
    # Let's build the constraint matrix
    # We'll collect all inequality constraints
    
    A_rows = []
    b_vals = []
    
    # IR constraints: p[t] - v1[t]*a[t][0] - v2[t]*a[t][1] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[2 * n_types + t] = 1  # p[t]
        row[t * n_goods + 0] = -support_list[t][0]  # -v1[t] * a[t][0]
        row[t * n_goods + 1] = -support_list[t][1]  # -v2[t] * a[t][1]
        A_rows.append(row)
        b_vals.append(0.0)
    
    # IC constraints: for all t, t': p[t] - v_t . a[t] <= p[t'] - v_t . a[t']
    # Rearranged: p[t] - p[t'] - v_t . a[t] + v_t . a[t'] <= 0
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[2 * n_types + t] = 1    # p[t]
            row[2 * n_types + t_prime] = -1  # -p[t']
            # -v_t . a[t]
            row[t * n_goods + 0] = -support_list[t][0]
            row[t * n_goods + 1] = -support_list[t][1]
            # +v_t . a[t']
            row[t_prime * n_goods + 0] = support_list[t][0]
            row[t_prime * n_goods + 1] = support_list[t][1]
            A_rows.append(row)
            b_vals.append(0.0)
    
    # Convert to arrays
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: 0 <= a[t][g] <= 1, 0 <= p[t] (no upper bound unless specified)
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for t in range(n_types):
        if max_payment is not None:
            bounds.append((0.0, max_payment))
        else:
            bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12}
        )
        
        if not result.success:
            # Fallback: return zero allocations and zero payments
            allocations = np.zeros((n_types, n_goods), dtype=np.float64)
            payments = np.zeros(n_types, dtype=np.float64)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x
        
        # Extract allocations and payments
        allocations = np.zeros((n_types, n_goods), dtype=np.float64)
        payments = np.zeros(n_types, dtype=np.float64)
        
        for t in range(n_types):
            for g in range(n_goods):
                allocations[t, g] = max(0.0, min(1.0, x[t * n_goods + g]))
            payments[t] = max(0.0, x[2 * n_types + t])
        
        # Clip small negative values to zero
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.clip(payments, 0.0, None)
        
        # Verify constraints and fix if needed
        # Check IR
        for t in range(n_types):
            utility = support_list[t][0] * allocations[t, 0] + support_list[t][1] * allocations[t, 1] - payments[t]
            if utility < -1e-8:
                # Fix by reducing payment
                payments[t] = max(0.0, support_list[t][0] * allocations[t, 0] + support_list[t][1] * allocations[t, 1])
        
        # Check IC: regret <= 1e-8
        max_regret = 0.0
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                # Utility of type t when reporting t'
                util_report = support_list[t][0] * allocations[t_prime, 0] + support_list[t][1] * allocations[t_prime, 1] - payments[t_prime]
                util_truth = support_list[t][0] * allocations[t, 0] + support_list[t][1] * allocations[t, 1] - payments[t]
                regret = util_report - util_truth
                if regret > max_regret:
                    max_regret = regret
        
        if max_regret > 1e-8:
            # IC violated, fall back to zero mechanism
            allocations = np.zeros((n_types, n_goods), dtype=np.float64)
            payments = np.zeros(n_types, dtype=np.float64)
        
        return {'allocations': allocations, 'payments': payments}
        
    except Exception as e:
        # Fallback: return zero allocations and zero payments
        allocations = np.zeros((n_types, n_goods), dtype=np.float64)
        payments = np.zeros(n_types, dtype=np.float64)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
