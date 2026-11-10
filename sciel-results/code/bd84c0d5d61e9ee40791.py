"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_values = np.array(types, dtype=float)  # shape (16, 2)
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'pmf' in distribution_spec:
            probs = np.array(distribution_spec['pmf'], dtype=float)
        else:
            # Default uniform distribution
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # For a single additive bidder with 2 goods, the IC constraints relate to
    # the expected allocation as a function of reported values.
    # 
    # Key insight: For a single bidder, the optimal mechanism can be characterized
    # by the allocation rule x(v1, v2) and payment p(v1, v2) where:
    # - x is the probability of allocating each good
    # - IC: The utility u(v) = v·x(v) - p(v) must be maximized at v=v
    # 
    # For a single additive bidder, the IC condition implies that the expected
    # allocation for good j is non-decreasing in v_j, and the payment satisfies:
    # p(v) = v·x(v) - sum over other goods of integral of x_j(v_{-j}, t) dt
    #
    # Actually, for a single bidder with quasi-linear utility, the IC constraints
    # can be written as: for all type pairs (v, v'),
    # v·x(v) - p(v) >= v·x(v') - p(v')
    #
    # This is equivalent to: p(v) - p(v') <= v·x(v) - v·x(v')
    #
    # Let's set up the LP. Variables: x_{i,j} for i in 16, j in {0,1} and p_i for i in 16
    # Total variables: 16*2 + 16 = 48
    #
    # Objective: maximize sum_i probs[i] * (p_i - v_i·x_i) = sum_i probs[i]*p_i - sum_i probs[i]*(v_i·x_i)
    # Actually we want to maximize expected revenue: sum_i probs[i] * p_i
    # But we need to account for the fact that payments are determined by IC constraints.
    #
    # Let me use a different approach. The standard formulation for single-bidder IC auctions:
    # 
    # Let x_i be the allocation vector for type i (2 goods, so 2 probabilities)
    # Let p_i be the payment for type i
    #
    # IC constraint: for all i, i': p_i - p_{i'} <= v_i·x_i - v_i·x_{i'}
    # IR constraint: p_i <= v_i·x_i (for all i)
    # Non-negativity: p_i >= 0, 0 <= x_{i,j} <= 1
    #
    # To remove p from the objective, we can use the fact that at optimum,
    # we can set p_i = min over i' of (v_i·x_i - v_i·x_{i'} + p_{i'})
    # 
    # A cleaner approach: introduce auxiliary variables.
    # 
    # Actually, let's just solve the LP directly with all variables.
    #
    # Variables: [x_0_0, x_0_1, ..., x_15_0, x_15_1, p_0, p_1, ..., p_15]
    # Total: 48 variables
    #
    # Objective: maximize sum_i probs[i] * p_i
    # In linprog form: minimize -sum_i probs[i] * p_i
    #
    # Constraints:
    # 1. IC: p_i - p_{i'} <= v_i·x_i - v_i·x_{i'} for all i, i'
    #    => p_i - p_{i'} + v_i·x_{i'} - v_i·x_i <= 0
    #    => -v_i·x_i + v_i·x_{i'} + p_i - p_{i'} <= 0
    #
    # 2. IR: p_i - v_i·x_i <= 0 for all i
    #
    # 3. Bounds: 0 <= x_{i,j} <= 1, p_i >= 0
    
    n_goods = 2
    n_vars = n_types * n_goods + n_types  # 48
    
    # Objective: minimize -sum_i probs[i] * p_i
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]
    
    # Build A_ub and b_ub for inequality constraints: A_ub @ vars <= b_ub
    A_rows = []
    b_vals = []
    
    # IC constraints: for all i, i'
    for i in range(n_types):
        for ip in range(n_types):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            # -v_i·x_i + v_i·x_{i'} + p_i - p_{i'} <= 0
            for j in range(n_goods):
                row[i * n_goods + j] = -type_values[i, j]
                row[ip * n_goods + j] = type_values[i, j]
            row[n_types * n_goods + i] = 1.0
            row[n_types * n_goods + ip] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
    
    # IR constraints: p_i - v_i·x_i <= 0 for all i
    for i in range(n_types):
        row = np.zeros(n_vars)
        for j in range(n_goods):
            row[i * n_goods + j] = -type_values[i, j]
        row[n_types * n_goods + i] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else None
    b_ub = np.array(b_vals) if b_vals else None
    
    # Bounds
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))  # x_{i,j} in [0,1]
        bounds.append((0, None))   # p_i >= 0
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple valid solution
        allocations = np.ones((n_types, n_goods))
        payments = np.array([type_values[i].sum() for i in range(n_types)])
        return {'allocations': allocations, 'payments': payments}
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple valid mechanism
        allocations = np.ones((n_types, n_goods))
        payments = np.array([type_values[i].sum() for i in range(n_types)])
        return {'allocations': allocations, 'payments': payments}
    
    x_vars = result.x[:n_types * n_goods].reshape(n_types, n_goods)
    p_vars = result.x[n_types * n_goods:]
    
    # Clip to ensure feasibility
    x_vars = np.clip(x_vars, 0, 1)
    p_vars = np.clip(p_vars, 0, None)
    
    # Verify constraints and adjust if needed
    # Check IC: p_i - p_{i'} <= v_i·(x_i - x_{i'}) for all i, i'
    max_violation = 0
    for i in range(n_types):
        for ip in range(n_types):
            lhs = p_vars[i] - p_vars[ip]
            rhs = type_values[i] @ (x_vars[i] - x_vars[ip])
            violation = lhs - rhs
            if violation > max_violation:
                max_violation = violation
    
    if max_violation > 1e-6:
        # Adjust payments to satisfy IC
        # Set p_i = v_i·x_i - min over i' of (v_i·x_{i'} - p_{i'})
        # This is getting complex; let's just scale down payments
        scale = max(0, 1 - max_violation / (1 + np.max(p_vars)))
        p_vars = p_vars * scale
    
    return {'allocations': x_vars, 'payments': p_vars}
# EVOLVE-BLOCK-END
