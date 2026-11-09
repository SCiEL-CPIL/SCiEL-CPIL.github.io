"""Rewritten program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Fallback result
    def fallback():
        return {
            'allocations': np.zeros((16, 2)),
            'payments': np.zeros(16)
        }
    
    # Try to get baseline as a fallback
    baseline_allocations = None
    baseline_payments = None
    try:
        from autoresearch_suite.science_economics_expansion import auction_baseline
        baseline_result = auction_baseline(distribution_spec, support, constraints)
        baseline_allocations = np.array(baseline_result['allocations'], dtype=float)
        baseline_payments = np.array(baseline_result['payments'], dtype=float)
    except:
        pass
    
    def return_baseline_or_fallback():
        if baseline_allocations is not None:
            return {
                'allocations': baseline_allocations,
                'payments': baseline_payments
            }
        return fallback()
    
    # Parse the distribution spec
    values = None
    probs = None
    if isinstance(distribution_spec, dict):
        probs = distribution_spec.get('probs', None)
        values = distribution_spec.get('values', None)
        if values is None:
            values = distribution_spec.get('support', None)
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 2:
            first, second = distribution_spec
            if isinstance(first, (list, tuple, np.ndarray)) and isinstance(second, (list, tuple, np.ndarray)):
                values, probs = first, second
            else:
                probs = distribution_spec
                values = None
        else:
            probs = distribution_spec
            values = None
    else:
        probs = None
        values = None
        
    # Default support: all 16 combinations of {0,1,2,3}^2
    if support is None:
        support = list(product([0, 1, 2, 3], repeat=2))
        
    # Ensure support is a list of tuples
    if not isinstance(support, (list, tuple)):
        support = list(support)
        
    n_types = len(support)
    
    # If we can't parse properly, return baseline or fallback
    if probs is None or n_types == 0:
        return return_baseline_or_fallback()
        
    # Convert to numpy
    probs = np.array(probs, dtype=float)
    if len(probs) != n_types:
        return return_baseline_or_fallback()
        
    # Ensure probs sum to 1 (normalize if needed)
    if probs.sum() > 1e-10:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
        
    # Extract value vectors
    # support is list of (v1, v2) for each type
    V = np.array(support, dtype=float)  # shape (n_types, 2)
    if V.shape[0] != n_types:
        return return_baseline_or_fallback()
    
    # Check time before solving
    if time.time() > deadline - 5.0:
        return return_baseline_or_fallback()
    
    # LP formulation with payment elimination:
    # Variables: x_{i,k} for i in 0..n_types-1, k in 0,1 (32 vars)
    #            u_i for i in 0..n_types-1 (16 vars)
    # Total: 48 variables
    #
    # Payment identity from IC: p_i = v_i . x_i - u_i
    #
    # Objective: maximize sum_i probs[i] * p_i = sum_i probs[i] * (v_i . x_i - u_i)
    #
    # Constraints:
    # 1. 0 <= x_{i,k} <= 1
    # 2. IR: u_i >= 0 (since u_i = v_i . x_i - p_i and IR says v_i . x_i - p_i >= 0)
    # 3. Nonneg payments: p_i = v_i . x_i - u_i >= 0 => u_i <= v_i . x_i
    # 4. IC: u_i >= v_i . x_j - p_j = v_i . x_j - (v_j . x_j - u_j) for all i != j
    #    => u_i - u_j >= v_i . x_j - v_j . x_j for all i != j
    #    => -u_i + u_j <= v_j . x_j - v_i . x_j for all i != j
    
    n_x = n_types * 2
    n_u = n_types
    n_vars = n_x + n_u
    
    # Objective: maximize sum_i probs[i] * (v_i . x_i - u_i)
    # = sum_i probs[i] * (V[i,0]*x_{i,0} + V[i,1]*x_{i,1} - u_i)
    # In linprog (minimize), negate:
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[i * 2] = -probs[i] * V[i, 0]
        c[i * 2 + 1] = -probs[i] * V[i, 1]
        c[n_x + i] = probs[i]  # +probs[i] * u_i in minimization
    
    # Inequality constraints: A_ub @ vars <= b_ub
    # 
    # Nonneg payments: u_i <= v_i . x_i  =>  -v_i . x_i + u_i <= 0
    # For each i: -V[i,0] at x_{i,0}, -V[i,1] at x_{i,1}, +1 at u_i
    #
    # IC: -u_i + u_j <= v_j . x_j - v_i . x_j for all i != j
    #     => -V[j,0]*x_{j,0} - V[j,1]*x_{j,1} + V[i,0]*x_{j,0} + V[i,1]*x_{j,1} - u_i + u_j <= 0
    #     Wait, let me re-derive:
    #     u_i >= v_i . x_j - p_j
    #     p_j = v_j . x_j - u_j
    #     u_i >= v_i . x_j - v_j . x_j + u_j
    #     u_i - u_j >= v_i . x_j - v_j . x_j
    #     -u_i + u_j <= v_j . x_j - v_i . x_j
    #     For fixed (i,j) with i != j:
    #       x_{j,0} coeff: V[j,0] - V[i,0]
    #       x_{j,1} coeff: V[j,1] - V[i,1]
    #       u_i coeff: -1
    #       u_j coeff: +1
    #       RHS: 0
    
    n_nn_pay = n_types
    n_ic = n_types * (n_types - 1)
    n_constraints = n_nn_pay + n_ic
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # Nonnegative payment constraints: u_i <= v_i . x_i
    for i in range(n_types):
        A_ub[i, i * 2] = -V[i, 0]
        A_ub[i, i * 2 + 1] = -V[i, 1]
        A_ub[i, n_x + i] = 1.0
        b_ub[i] = 0.0
    
    # IC constraints
    row = n_nn_pay
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            # -u_i + u_j <= v_j . x_j - v_i . x_j
            # Coefficients on x_{j,0}: V[j,0] - V[i,0]
            # Coefficients on x_{j,1}: V[j,1] - V[i,1]
            # Coefficients on u_i: -1
            # Coefficients on u_j: +1
            A_ub[row, j * 2] = V[j, 0] - V[i, 0]
            A_ub[row, j * 2 + 1] = V[j, 1] - V[i, 1]
            A_ub[row, n_x + i] = -1.0
            A_ub[row, n_x + j] = 1.0
            b_ub[row] = 0.0
            row += 1
    
    # Bounds: 0 <= x_{i,k} <= 1, u_i >= 0
    bounds = []
    for i in range(n_types):
        for k in range(2):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status != 0:
            return return_baseline_or_fallback()
        
        x_opt = result.x
        allocations = x_opt[:n_x].reshape(n_types, 2)
        utils = x_opt[n_x:]
        
        # Check time
        if time.time() > deadline - 2.0:
            return return_baseline_or_fallback()
        
        # Recover payments: p_i = v_i . x_i - u_i
        payments = np.sum(V * allocations, axis=1) - utils
        
        # Verify constraints with tolerance
        tol = 1e-6
        
        # Check allocations in [0, 1]
        if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
            return return_baseline_or_fallback()
        allocations = np.clip(allocations, 0.0, 1.0)
        
        # Check nonnegative payments
        if np.any(payments < -tol):
            return return_baseline_or_fallback()
        payments = np.maximum(payments, 0.0)
        
        # Check IR: v_i . x_i - p_i >= 0 (should be utils >= 0)
        if np.any(utils < -tol):
            return return_baseline_or_fallback()
        
        # Check IC: for all i, j, v_i . x_i - p_i >= v_i . x_j - p_j
        # i.e., utils[i] >= v_i . x_j - payments[j]
        ic_violation = False
        for i in range(n_types):
            for j in range(n_types):
                dev_util = np.dot(V[i], allocations[j]) - payments[j]
                if utils[i] < dev_util - tol:
                    ic_violation = True
                    break
            if ic_violation:
                break
        
        if ic_violation:
            return return_baseline_or_fallback()
        
        return {
            'allocations': allocations,
            'payments': payments
        }
        
    except Exception as e:
        return return_baseline_or_fallback()
# EVOLVE-BLOCK-END
