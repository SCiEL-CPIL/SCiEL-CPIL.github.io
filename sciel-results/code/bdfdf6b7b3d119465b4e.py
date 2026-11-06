"""Incentive-compatible finite auction design via direct LP solve."""
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
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        elif 'values' in distribution_spec:
            probs = np.array(distribution_spec['values'], dtype=float)
        else:
            # Try to find any list-like value
            for v in distribution_spec.values():
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 16:
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    
    # Parse support
    if support is None:
        support_list = [(i, j) for i in range(4) for j in range(4)]
    elif isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i, j) for i in range(4) for j in range(4)]
    
    # Ensure we have exactly 16 types
    if len(support_list) != 16:
        if len(support_list) < 16:
            # Pad with remaining combinations
            existing = set(support_list)
            all_combos = [(i, j) for i in range(4) for j in range(4)]
            for combo in all_combos:
                if len(support_list) >= 16:
                    break
                if combo not in existing:
                    support_list.append(combo)
            # If still short, duplicate last
            while len(support_list) < 16:
                support_list.append(support_list[-1])
            # Pad probs
            if len(probs) < 16:
                probs = np.concatenate([probs, np.zeros(16 - len(probs))])
        else:
            support_list = support_list[:16]
            probs = probs[:16]
    
    n_types = 16
    values = np.array(support_list, dtype=float)  # shape (16, 2)
    v1 = values[:, 0]
    v2 = values[:, 1]
    
    # Decision variables:
    # x[t, g] = probability of allocating good g to bidder when type is t
    # p[t] = payment when type is t
    # 
    # Total variables: 16*2 + 16 = 48
    
    n_x = n_types * 2  # 32
    n_p = n_types  # 16
    n_vars = n_x + n_p  # 48
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # In linprog form (minimize), we minimize -sum_t probs[t] * p[t]
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    # IR constraints: for each t, -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    # IC constraints: for each pair (t, t'), t != t'
    # -v1[t]*x[t,0] - v2[t]*x[t,1] + v1[t]*x[t',0] + v2[t]*x[t',1] + p[t] - p[t'] <= 0
    
    # Total constraints: 16 (IR) + 16*15 (IC, excluding t==t') = 16 + 240 = 256
    n_constraints = n_types + n_types * (n_types - 1)
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # IR constraints
    for t in range(n_types):
        row = t
        A_ub[row, t] = -v1[t]      # coefficient of x[t, 0]
        A_ub[row, t + n_types] = -v2[t]  # coefficient of x[t, 1]
        A_ub[row, n_x + t] = 1.0   # coefficient of p[t]
        b_ub[row] = 0.0
    
    # IC constraints
    constraint_idx = n_types
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = constraint_idx
            constraint_idx += 1
            # -v1[t]*x[t,0] - v2[t]*x[t,1] + v1[t]*x[tp,0] + v2[t]*x[tp,1] + p[t] - p[tp] <= 0
            A_ub[row, t] = -v1[t]
            A_ub[row, t + n_types] = -v2[t]
            A_ub[row, tp] = v1[t]
            A_ub[row, tp + n_types] = v2[t]
            A_ub[row, n_x + t] = 1.0
            A_ub[row, n_x + tp] = -1.0
            b_ub[row] = 0.0
    
    # Bounds: 0 <= x[t,g] <= 1, 0 <= p[t] (nonnegative payments)
    bounds = [(0, 1)] * n_x + [(0, None)] * n_p
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a simple mechanism
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x_sol = result.x[:n_x].reshape(n_types, 2)
        p_sol = result.x[n_x:]
        
        # Ensure numerical precision
        x_sol = np.clip(x_sol, 0, 1)
        p_sol = np.maximum(p_sol, 0)
        
        # Check time
        if time.time() > deadline:
            return {'allocations': x_sol, 'payments': p_sol}
        
        # Verify constraints with tolerance
        max_violation = 0.0
        for t in range(n_types):
            # IR
            util_t = v1[t]*x_sol[t, 0] + v2[t]*x_sol[t, 1] - p_sol[t]
            max_violation = max(max_violation, -util_t)
            
            # IC
            for tp in range(n_types):
                if t == tp:
                    continue
                util_tp = v1[t]*x_sol[tp, 0] + v2[t]*x_sol[tp, 1] - p_sol[tp]
                violation = util_t - util_tp
                max_violation = max(max_violation, -violation)
        
        if max_violation > 1e-8:
            # Try to fix small violations by adjusting payments
            for t in range(n_types):
                # IR: ensure p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1]
                max_pay = v1[t]*x_sol[t, 0] + v2[t]*x_sol[t, 1]
                if p_sol[t] > max_pay:
                    p_sol[t] = max_pay
                # p[t] >= 0
                p_sol[t] = max(p_sol[t], 0)
            
            # Re-verify
            max_violation = 0.0
            for t in range(n_types):
                util_t = v1[t]*x_sol[t, 0] + v2[t]*x_sol[t, 1] - p_sol[t]
                max_violation = max(max_violation, -util_t)
                for tp in range(n_types):
                    if t == tp:
                        continue
                    util_tp = v1[t]*x_sol[tp, 0] + v2[t]*x_sol[tp, 1] - p_sol[tp]
                    violation = util_t - util_tp
                    max_violation = max(max_violation, -violation)
            
            if max_violation > 1e-8:
                # Fallback to a simple valid mechanism
                # Mechanism: Allocate good g if value >= 1, pay 1.
                alloc_fallback = np.zeros((n_types, 2))
                pay_fallback = np.zeros(n_types)
                for t in range(n_types):
                    if v1[t] >= 1.0:
                        alloc_fallback[t, 0] = 1.0
                        pay_fallback[t] += 1.0
                    if v2[t] >= 1.0:
                        alloc_fallback[t, 1] = 1.0
                        pay_fallback[t] += 1.0
                
                # Verify fallback
                max_violation_fb = 0.0
                for t in range(n_types):
                    util_t = v1[t]*alloc_fallback[t, 0] + v2[t]*alloc_fallback[t, 1] - pay_fallback[t]
                    max_violation_fb = max(max_violation_fb, -util_t)
                    for tp in range(n_types):
                        if t == tp:
                            continue
                        util_tp = v1[t]*alloc_fallback[tp, 0] + v2[t]*alloc_fallback[tp, 1] - pay_fallback[tp]
                        violation = util_t - util_tp
                        max_violation_fb = max(max_violation_fb, -violation)
                
                if max_violation_fb <= 1e-8:
                    return {'allocations': alloc_fallback, 'payments': pay_fallback}
                else:
                    # Last resort: Zero mechanism
                    return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
        
        return {'allocations': x_sol, 'payments': p_sol}
    
    except Exception as e:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
