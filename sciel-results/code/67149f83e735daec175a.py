"""Auction design exploiting single-bidder structure via monotone allocation decomposition."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                probs = np.array([dist.get(t, 0.0) for t in types], dtype=float)
            elif isinstance(dist, (list, tuple)):
                probs = np.array(dist, dtype=float)
            else:
                probs = np.array([0.5/16]*16, dtype=float)
        else:
            probs = np.array([1.0/16]*16, dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.array([1.0/16]*16, dtype=float)
    
    # Ensure probabilities sum to 1
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.array([1.0/16]*16, dtype=float)
    
    # Parse constraints
    if constraints is not None and isinstance(constraints, dict):
        regret_tol = constraints.get('regret_tol', 1e-8)
        ir_tol = constraints.get('ir_tol', 1e-8)
    else:
        regret_tol = 1e-8
        ir_tol = 1e-8
    
    # For a single additive bidder with 2 goods, the problem decomposes.
    # The joint allocation (x1, x2) must be monotone in both dimensions for IC.
    # We can use the fact that for each good independently, the allocation rule
    # must be non-decreasing in the bidder's value for that good.
    # 
    # Key insight: For a single bidder, IC + IR + non-negative payments
    # is equivalent to finding a monotone allocation rule x(v) and setting
    # payments p(v) = v·x(v) - u(v) where u(v) is the utility function
    # determined by the monotone structure.
    #
    # Actually, the cleanest approach for finite types:
    # The IC constraints are: u(t) >= u(t') + (v(t) - v(t'))·x(t') for all t, t'
    # where u(t) = v(t)·x(t) - p(t)
    #
    # This means: u(t) - v(t)·x(t) >= u(t') - v(t)·x(t') for all t, t'
    # => -p(t) >= u(t') - v(t)·x(t') for all t, t'
    # => p(t) <= v(t)·x(t') - u(t') for all t, t'
    #
    # Also IR: u(t) >= 0 for all t
    # And p(t) >= 0 for all t
    #
    # To maximize revenue, we want to maximize sum_t probs[t] * p(t)
    #
    # The binding constraints for p(t) are:
    # p(t) <= min_{t'} [v(t)·x(t') - u(t')]
    # p(t) >= 0
    # u(t) = v(t)·x(t) - p(t) >= 0
    #
    # This is still complex. Let's use a different approach.
    #
    # For a single bidder, the optimal mechanism can be characterized by:
    # 1. Choose allocation probabilities x1[t], x2[t] in [0,1] that are monotone
    #    (x1[t] is non-decreasing in v1[t], x2[t] is non-decreasing in v2[t])
    # 2. Set payments to maximize revenue subject to IC and IR.
    #
    # The payment for type t is: p(t) = v(t)·x(t) - u(t)
    # where u(t) is the utility that makes the mechanism IC and IR.
    #
    # For finite types, u(t) is determined by the "critical type" structure.
    # Specifically, u(t) = max_{t' < t} [u(t') + (v(t) - v(t'))·x(t')]
    # where t' < t means t' is "below" t in the partial order.
    #
    # Actually, let's just solve the LP but with a smarter formulation.
    # The key is that the IC constraints can be simplified.
    #
    # For the finite grid {0,1,2,3}^2, the IC constraints only need to be
    # checked along the "adjacent" types in the grid (like in a lattice).
    # Specifically, for a type (v1, v2), the critical types are:
    #   (v1-1, v2), (v1, v2-1) if they exist
    #
    # This reduces the number of constraints significantly.
    
    # Let's use the LP approach but with reduced constraints.
    # Variables: x1[0..15], x2[0..15], p[0..15] => 48 variables
    
    n_vars = 48
    x1_offset = 0
    x2_offset = 16
    p_offset = 32
    
    # Objective: maximize sum_t probs[t] * p(t)
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_offset + t] = -probs[t]
    
    # Build constraints
    A_ub_rows = []
    b_ub_vals = []
    
    # For IC, we only need to check adjacent types in the grid.
    # For type (v1, v2), check against (v1-1, v2) and (v1, v2-1)
    for t in range(n_types):
        v1_t, v2_t = types[t]
        
        # Check against (v1-1, v2)
        if v1_t > 0:
            t2 = type_to_idx[(v1_t - 1, v2_t)]
            row = np.zeros(n_vars)
            row[x1_offset + t] = -v1_t
            row[x2_offset + t] = -v2_t
            row[p_offset + t] = 1
            row[x1_offset + t2] = v1_t
            row[x2_offset + t2] = v2_t
            row[p_offset + t2] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
        
        # Check against (v1, v2-1)
        if v2_t > 0:
            t2 = type_to_idx[(v1_t, v2_t - 1)]
            row = np.zeros(n_vars)
            row[x1_offset + t] = -v1_t
            row[x2_offset + t] = -v2_t
            row[p_offset + t] = 1
            row[x1_offset + t2] = v1_t
            row[x2_offset + t2] = v2_t
            row[p_offset + t2] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
    
    # IR constraints
    for t in range(n_types):
        row = np.zeros(n_vars)
        v1_t, v2_t = types[t]
        row[x1_offset + t] = -v1_t
        row[x2_offset + t] = -v2_t
        row[p_offset + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1[t]
    for t in range(n_types):
        bounds.append((0, 1))  # x2[t]
    for t in range(n_types):
        bounds.append((0, None))  # p[t]
    
    # Use scipy's linprog
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple feasible solution
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_opt = result.x
    
    allocations = np.zeros((n_types, 2))
    for t in range(n_types):
        allocations[t, 0] = x_opt[x1_offset + t]
        allocations[t, 1] = x_opt[x2_offset + t]
    
    payments = np.zeros(n_types)
    for t in range(n_types):
        payments[t] = x_opt[p_offset + t]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    # Verify and fix constraints
    for t in range(n_types):
        v1_t, v2_t = types[t]
        utility = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if utility < -1e-10:
            payments[t] = max(0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
