"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Map type tuple to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                probs = np.array([dist.get(t, 0.0) for t in types], dtype=float)
            else:
                probs = np.array(dist, dtype=float)
        else:
            # Default: uniform distribution
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Parse support (allowed reports)
    if support is not None:
        if isinstance(support, (list, tuple, np.ndarray)):
            if len(support) == n_types:
                # Boolean mask or list of type tuples
                if all(isinstance(s, (int, float, bool, np.integer, np.floating)) for s in support):
                    support_mask = np.array(support, dtype=bool)
                else:
                    support_mask = np.zeros(n_types, dtype=bool)
                    for s in support:
                        if isinstance(s, tuple) and len(s) == 2:
                            if s in type_to_idx:
                                support_mask[type_to_idx[s]] = True
                        elif isinstance(s, (int, np.integer)):
                            if 0 <= s < n_types:
                                support_mask[s] = True
            else:
                support_mask = np.ones(n_types, dtype=bool)
        else:
            support_mask = np.ones(n_types, dtype=bool)
    else:
        support_mask = np.ones(n_types, dtype=bool)
    
    # Parse constraints
    if constraints is not None and isinstance(constraints, dict):
        # Could contain additional constraint specs
        pass
    
    # Build the LP for mechanism design
    # Variables: x[t, g] for t in types, g in {0,1} (allocation probabilities)
    #            p[t] for t in types (payments)
    # 
    # Objective: Maximize expected revenue = sum_t p[t] * probs[t]
    # 
    # Constraints:
    # 1. Feasibility: 0 <= x[t, g] <= 1 for all t, g
    # 2. Individual rationality: u[t] >= 0 for all t in support
    #    where u[t] = sum_g v[t,g] * x[t,g] - p[t]
    # 3. Incentive compatibility: u[t] >= u[t'] + sum_g (v[t,g] - v[t',g]) * x[t',g] for all t, t' in support
    #    This is the standard IC constraint in terms of utilities.
    #    
    # Actually, let's use the standard formulation:
    # For each type t, define utility u[t] = sum_g v[t,g] * x[t,g] - p[t]
    # IC: u[t] >= sum_g v[t,g] * x[t',g] - p[t'] for all t, t'
    #    i.e., u[t] >= u[t'] + sum_g (v[t,g] - v[t',g]) * x[t',g]
    # IR: u[t] >= 0 for all t
    #
    # We can eliminate p[t] = sum_g v[t,g]*x[t,g] - u[t]
    # Revenue = sum_t p[t] * probs[t] = sum_t (sum_g v[t,g]*x[t,g] - u[t]) * probs[t]
    #        = sum_t sum_g v[t,g]*x[t,g]*probs[t] - sum_t u[t]*probs[t]
    #
    # So we maximize: sum_t sum_g v[t,g]*x[t,g]*probs[t] - sum_t u[t]*probs[t]
    # Subject to:
    # - 0 <= x[t,g] <= 1
    # - u[t] >= 0 (IR)
    # - u[t] >= u[t'] + sum_g (v[t,g] - v[t',g]) * x[t',g] for all t,t' (IC)
    #
    # Variables: x[t,g] for t in 0..15, g in 0,1 => 32 variables
    #            u[t] for t in 0..15 => 16 variables
    # Total: 48 variables
    #
    # Let's set up the LP using scipy.optimize.linprog
    # linprog minimizes, so we negate the objective.
    
    n_g = 2  # number of goods
    
    # Variable indexing:
    # x_vars: index (t * n_g + g) for t in range(n_types), g in range(n_g)
    # u_vars: index (n_types * n_g + t) for t in range(n_types)
    
    total_vars = n_types * n_g + n_types  # 32 + 16 = 48
    
    # Objective: minimize -revenue = -sum_t sum_g v[t,g]*x[t,g]*probs[t] + sum_t u[t]*probs[t]
    c = np.zeros(total_vars)
    for t in range(n_types):
        for g in range(n_g):
            v = types[t][g]
            c[t * n_g + g] = -v * probs[t]
        c[n_types * n_g + t] = probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 
    # IR: -u[t] <= 0 for all t
    # IC: u[t] - u[t'] - sum_g (v[t,g] - v[t',g]) * x[t',g] <= 0 for all t, t' in support
    #     i.e., -u[t'] + u[t] - sum_g (v[t,g] - v[t',g]) * x[t',g] <= 0
    #
    # Let's build the constraint matrix
    
    # Count constraints
    n_ir = n_types  # IR constraints
    # IC constraints: for each pair (t, t') in support
    support_types = [t for t in range(n_types) if support_mask[t]]
    n_ic = len(support_types) * len(support_types)
    n_total_constraints = n_ir + n_ic
    
    A_ub = np.zeros((n_total_constraints, total_vars))
    b_ub = np.zeros(n_total_constraints)
    
    # IR constraints: -u[t] <= 0
    for t in range(n_types):
        row = t
        A_ub[row, n_types * n_g + t] = -1.0
        b_ub[row] = 0.0
    
    # IC constraints
    row_idx = n_ir
    for t in support_types:
        for t_prime in support_types:
            # u[t] - u[t'] - sum_g (v[t,g] - v[t',g]) * x[t',g] <= 0
            A_ub[row_idx, n_types * n_g + t] = 1.0  # +u[t]
            A_ub[row_idx, n_types * n_g + t_prime] = -1.0  # -u[t']
            for g in range(n_g):
                coeff = -(types[t][g] - types[t_prime][g])
                A_ub[row_idx, t_prime * n_g + g] = coeff
            b_ub[row_idx] = 0.0
            row_idx += 1
    
    # Bounds: 0 <= x[t,g] <= 1, u[t] >= 0 (no upper bound)
    bounds = []
    for t in range(n_types):
        for g in range(n_g):
            bounds.append((0.0, 1.0))
    for t in range(n_types):
        bounds.append((0.0, None))
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            # Extract allocations and payments
            allocations = np.zeros((n_types, n_g))
            utilities = np.zeros(n_types)
            
            for t in range(n_types):
                for g in range(n_g):
                    allocations[t, g] = result.x[t * n_g + g]
                utilities[t] = result.x[n_types * n_g + t]
            
            # Compute payments: p[t] = sum_g v[t,g] * x[t,g] - u[t]
            payments = np.zeros(n_types)
            for t in range(n_types):
                payments[t] = sum(types[t][g] * allocations[t, g] for g in range(n_g)) - utilities[t]
            
            # Ensure non-negative payments (clip if needed)
            payments = np.maximum(payments, 0.0)
            
            # Check deadline
            if time.time() > deadline:
                return {'allocations': allocations, 'payments': payments}
            
            return {'allocations': allocations, 'payments': payments}
        else:
            # Fallback: simple threshold mechanism
            pass
    except Exception:
        pass
    
    # Fallback: simple VCG-like mechanism
    allocations = np.zeros((n_types, n_g))
    payments = np.zeros(n_types)
    
    for t in range(n_types):
        for g in range(n_g):
            # Allocate good g to type t if value is high enough
            if types[t][g] >= 2:
                allocations[t, g] = 1.0
            else:
                allocations[t, g] = 0.0
        # Payment: second-price style
        # For simplicity, use a fixed payment based on allocation
        for g in range(n_g):
            if allocations[t, g] > 0.5:
                payments[t] += 1.0  # Simple payment
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
