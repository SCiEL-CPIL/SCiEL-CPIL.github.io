import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    
    # 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    
    # Parse distribution
    probs = np.ones(n_types) / n_types
    if isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
            probs = probs / probs.sum()
        elif isinstance(distribution_spec, dict):
            for i, (v1, v2) in enumerate(types):
                if (v1, v2) in distribution_spec:
                    probs[i] = distribution_spec[(v1, v2)]
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, str):
        if 'uniform' in distribution_spec.lower() or 'independent' in distribution_spec.lower():
            probs = np.ones(n_types) / n_types
        elif 'product' in distribution_spec.lower():
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    
    # Ensure valid probabilities
    probs = np.clip(probs, 0, 1)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Decision variables: x[0:16] = alloc good 1, x[16:32] = alloc good 2, x[32:48] = payments
    n_vars = n_types * 3  # 48
    
    # Objective: maximize expected revenue
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]
    
    A_ub = []
    b_ub = []
    
    # IR constraints: p(i) - v1(i)*x1(i) - v2(i)*x2(i) <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = types[i]
        row[i] = -v1
        row[n_types + i] = -v2
        row[32 + i] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -v1_i
            row[n_types + i] = -v2_i
            row[32 + i] = 1
            row[j] = v1_i
            row[n_types + j] = v2_i
            row[32 + j] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Solve LP with tight tolerances
    options = {'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12}
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    
    if not result.success:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i, (v1, v2) in enumerate(types):
            if v1 + v2 >= 2:
                allocations[i] = [1, 1]
                payments[i] = 2.0
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = np.column_stack([x[0:n_types], x[n_types:2*n_types]])
    payments = x[32:48]
    
    # Verify and fix numerical issues
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify IC and IR constraints; if violated, solve a correction LP
    def check_constraints(alloc, pay):
        """Check all IR and IC constraints. Return list of violated constraint indices."""
        violations = []
        # IR: p(i) <= v1*x1 + v2*x2
        for i in range(n_types):
            v1, v2 = types[i]
            if pay[i] > v1 * alloc[i, 0] + v2 * alloc[i, 1] + 1e-8:
                violations.append(('ir', i, pay[i] - v1 * alloc[i, 0] - v2 * alloc[i, 1]))
        # IC
        for i in range(n_types):
            v1_i, v2_i = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                util_truth = v1_i * alloc[i, 0] + v2_i * alloc[i, 1] - pay[i]
                util_lie = v1_i * alloc[j, 0] + v2_i * alloc[j, 1] - pay[j]
                if util_lie > util_truth + 1e-8:
                    violations.append(('ic', i, j, util_lie - util_truth))
        return violations
    
    violations = check_constraints(allocations, payments)
    
    if violations and time.time() - start_time < 80:
        # Solve a small correction LP to fix violations
        # Minimize sum of violation amounts
        # Variables: same 48 variables + slack variables for each violation
        n_viol = len(violations)
        n_corr_vars = n_vars + n_viol
        
        c_corr = np.zeros(n_corr_vars)
        for k in range(n_viol):
            c_corr[n_vars + k] = 1.0  # minimize sum of slacks
        
        A_ub_corr = []
        b_ub_corr = []
        
        # Original constraints
        A_ub_corr.append(A_ub)
        b_ub_corr.extend(b_ub)
        
        # Violation constraints: for each violation, add constraint that ensures it's satisfied
        # For IR violation at type i: p(i) - v1*x1(i) - v2*x2(i) <= slack
        # For IC violation (i,j): -v1_i*x1(i) - v2_i*x2(i) + p(i) + v1_i*x1(j) + v2_i*x2(j) - p(j) <= slack
        
        for k, viol in enumerate(violations):
            row = np.zeros(n_corr_vars)
            if viol[0] == 'ir':
                i = viol[1]
                v1, v2 = types[i]
                row[i] = -v1
                row[n_types + i] = -v2
                row[32 + i] = 1
            else:
                i, j = viol[1], viol[2]
                v1_i, v2_i = types[i]
                row[i] = -v1_i
                row[n_types + i] = -v2_i
                row[32 + i] = 1
                row[j] = v1_i
                row[n_types + j] = v2_i
                row[32 + j] = -1
            row[n_vars + k] = 1  # + slack
            A_ub_corr.append(row)
            b_ub_corr.append(0)
        
        A_ub_corr = np.array(A_ub_corr)
        b_ub_corr = np.array(b_ub_corr)
        
        bounds_corr = bounds + [(0, None)] * n_viol
        
        result_corr = linprog(c_corr, A_ub=A_ub_corr, b_ub=b_ub_corr, bounds=bounds_corr, method='highs', options=options)
        
        if result_corr.success:
            x_corr = result_corr.x
            allocations_new = np.column_stack([x_corr[0:n_types], x_corr[n_types:2*n_types]])
            payments_new = x_corr[32:48]
            allocations_new = np.clip(allocations_new, 0, 1)
            payments_new = np.maximum(payments_new, 0)
            
            # Check if the correction improved things
            violations_new = check_constraints(allocations_new, payments_new)
            if len(violations_new) < len(violations):
                allocations = allocations_new
                payments = payments_new
    
    if time.time() - start_time > 85:
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': allocations, 'payments': payments}
