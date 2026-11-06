"""Incentive-compatible finite auction design via LP over utilities with monotonicity constraints."""
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
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=np.float64)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=np.float64)
        else:
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
    if isinstance(constraints, dict):
        max_payment = constraints.get('max_payment', None)
    
    n_types = 16
    n_goods = 2
    
    # LP over utilities u[t] and allocations a[t][g]
    # Variables: u[0:16] (utilities), a[16:32] (allocations)
    # Total: 32 variables
    # 
    # Objective: maximize sum_t probs[t] * (v_t . a[t] - u[t])
    #           = maximize sum_t probs[t] * v_t . a[t] - sum_t probs[t] * u[t]
    #
    # Constraints:
    # 1. u[t] >= 0 (IR)
    # 2. u[t] >= u[t'] + (v_t - v_t') . a[t'] for all t, t' (IC via monotonicity of utility)
    #    This is equivalent to: u[t] - u[t'] - (v_t - v_t') . a[t'] >= 0
    # 3. 0 <= a[t][g] <= 1
    # 4. a must be monotone: if v1[t] >= v1[t'] and v2[t] >= v2[t'], then a[t][g] >= a[t'][g]
    #    Actually, for single bidder with additive values, IC is equivalent to:
    #    - Allocation is monotone in each dimension
    #    - Payment = v_t . a[t] - u[t] where u is the "virtual utility"
    #
    # Better formulation: Use the fact that for a single bidder, a mechanism is truthful iff
    # the allocation rule is monotone and payments satisfy the envelope condition.
    #
    # Let's use: variables u[t], a[t][0], a[t][1]
    # Objective: max sum_t probs[t] * (v1[t]*a[t][0] + v2[t]*a[t][1] - u[t])
    # Subject to:
    #   u[t] >= 0
    #   u[t] - u[t'] >= (v1[t]-v1[t'])*a[t'][0] + (v2[t]-v2[t'])*a[t'][1] for all t,t'
    #   0 <= a[t][g] <= 1
    #
    # The IC constraint u[t] - u[t'] >= (v_t - v_t') . a[t'] ensures that reporting t' 
    # doesn't give higher utility than reporting truthfully for type t.
    
    n_vars = n_types + n_types * n_goods  # 16 + 32 = 48
    
    # Objective coefficients: minimize -sum_t probs[t]*(v1[t]*a[t][0] + v2[t]*a[t][1] - u[t])
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[t] = probs[t]  # coefficient for u[t] (we want to minimize -revenue, so +probs[t]*u[t])
        c[n_types + t * n_goods + 0] = -probs[t] * support_list[t][0]
        c[n_types + t * n_goods + 1] = -probs[t] * support_list[t][1]
    
    A_rows = []
    b_vals = []
    
    # IR: u[t] >= 0 => -u[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[t] = -1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    # IC: u[t] - u[t'] - (v1[t]-v1[t'])*a[t'][0] - (v2[t]-v2[t'])*a[t'][1] >= 0
    # => -u[t] + u[t'] + (v1[t]-v1[t'])*a[t'][0] + (v2[t]-v2[t'])*a[t'][1] <= 0
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[t] = -1.0
            row[t_prime] = 1.0
            dv1 = support_list[t][0] - support_list[t_prime][0]
            dv2 = support_list[t][1] - support_list[t_prime][1]
            row[n_types + t_prime * n_goods + 0] = dv1
            row[n_types + t_prime * n_goods + 1] = dv2
            A_rows.append(row)
            b_vals.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds
    bounds = []
    for t in range(n_types):
        bounds.append((0.0, None))  # u[t] >= 0
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    
    # Solve LP
    try:
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'presolve': True}
        )
        
        if not result.success:
            allocations = np.zeros((n_types, n_goods), dtype=np.float64)
            payments = np.zeros(n_types, dtype=np.float64)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x
        
        # Extract utilities and allocations
        utilities = x[:n_types].copy()
        allocations = np.zeros((n_types, n_goods), dtype=np.float64)
        for t in range(n_types):
            for g in range(n_goods):
                allocations[t, g] = max(0.0, min(1.0, x[n_types + t * n_goods + g]))
        
        # Compute payments: p[t] = v_t . a[t] - u[t]
        payments = np.zeros(n_types, dtype=np.float64)
        for t in range(n_types):
            payments[t] = support_list[t][0] * allocations[t, 0] + support_list[t][1] * allocations[t, 1] - utilities[t]
            payments[t] = max(0.0, payments[t])
        
        # Verify IC: regret <= 1e-8
        max_regret = 0.0
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                util_report = support_list[t][0] * allocations[t_prime, 0] + support_list[t][1] * allocations[t_prime, 1] - payments[t_prime]
                util_truth = support_list[t][0] * allocations[t, 0] + support_list[t][1] * allocations[t, 1] - payments[t]
                regret = util_report - util_truth
                if regret > max_regret:
                    max_regret = regret
        
        if max_regret > 1e-8:
            allocations = np.zeros((n_types, n_goods), dtype=np.float64)
            payments = np.zeros(n_types, dtype=np.float64)
        
        return {'allocations': allocations, 'payments': payments}
        
    except Exception as e:
        allocations = np.zeros((n_types, n_goods), dtype=np.float64)
        payments = np.zeros(n_types, dtype=np.float64)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
