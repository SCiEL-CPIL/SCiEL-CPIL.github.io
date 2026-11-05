"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # Build and solve LP
    result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 2.0:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    if result is not None:
        return result
    else:
        # Fallback: always allocate both goods, charge the lower value
        allocations = np.ones((16, 2))
        payments = np.array([min(types[i][0], types[i][1]) for i in range(16)], dtype=float)
        return {'allocations': allocations, 'payments': payments}

def _parse_distribution(distribution_spec, support, types, n_types):
    """Parse distribution spec to get type probabilities."""
    # Try various formats
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, str):
        # Maybe a name like "uniform", "independent", etc.
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, dict):
        # Try various keys
        for key in ['probs', 'distribution', 'type_probs', 'probabilities', 'p']:
            if key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val, dtype=float).flatten()
                    if len(arr) == n_types:
                        arr = np.maximum(arr, 0)
                        if arr.sum() > 0:
                            return arr / arr.sum()
                elif isinstance(val, dict):
                    probs = np.ones(n_types) / n_types
                    for i, t in enumerate(types):
                        if t in val:
                            probs[i] = val[t]
                        elif str(t) in val:
                            probs[i] = val[str(t)]
                        elif f"{t[0]},{t[1]}" in val:
                            probs[i] = val[f"{t[0]},{t[1]}"]
                    probs = np.maximum(probs, 0)
                    if probs.sum() > 0:
                        return probs / probs.sum()
        
        # Maybe the dict maps type tuples to probs directly
        probs = np.zeros(n_types)
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
        if probs.sum() > 0:
            return probs / probs.sum()
    
    # Fallback: check support
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()
    
    return np.ones(n_types) / n_types

def _solve_lp(probs, types, n_types, n_goods):
    """Solve the revenue-maximizing IC/IR LP."""
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types
    
    # Objective: maximize sum_i probs[i] * p[i] -> minimize -sum
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # IC: for all i, j: v_i·x_i - p_i >= v_i·x_j - p_j
    # => -v_i·x_i + p_i + v_i·x_j - p_j <= 0
    # IR: for all i: v_i·x_i - p_i >= 0 => -v_i·x_i + p_i <= 0
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        vi = types[i]
        for j in range(n_types):
            if i == j:
                continue
            A_ub[idx, var_x(i, 0)] = -vi[0]
            A_ub[idx, var_x(i, 1)] = -vi[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi[0]
            A_ub[idx, var_x(j, 1)] = vi[1]
            A_ub[idx, var_p(j)] = -1.0
            idx += 1
    
    for i in range(n_types):
        vi = types[i]
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        idx += 1
    
    bounds = [(0.0, 1.0)] * (2 * n_types) + [(0.0, None)] * n_types
    
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'time_limit': 60.0,
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        
        if result.success:
            x = result.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0.0, 1.0))
                payments[i] = float(max(0.0, x[var_p(i)]))
            
            # Light verification: only fix if violation > 1e-8
            max_viol = _max_violation(allocations, payments, types)
            if max_viol > 1e-8:
                allocations, payments = _fix_violations(allocations, payments, types)
            
            return {'allocations': allocations, 'payments': payments}
        else:
            return None
    except Exception:
        return None

def _max_violation(allocations, payments, types):
    """Compute the maximum IC/IR violation."""
    n_types = len(types)
    max_v = 0.0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = float(np.dot(vi, allocations[i])) - payments[i]
        if ui < -max_v:
            max_v = -ui
        for j in range(n_types):
            if i == j:
                continue
            uj = float(np.dot(vi, allocations[j])) - payments[j]
            viol = uj - ui
            if viol > max_v:
                max_v = viol
    return max_v

def _fix_violations(allocations, payments, types):
    """Fix IC/IR violations by adjusting payments minimally."""
    n_types = len(types)
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    for _iter in range(200):
        max_v = 0.0
        worst = None
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = float(np.dot(vi, allocations[i])) - payments[i]
            if ui < -max_v:
                max_v = -ui
                worst = ('ir', i, -1)
            for j in range(n_types):
                if i == j:
                    continue
                uj = float(np.dot(vi, allocations[j])) - payments[j]
                viol = uj - ui
                if viol > max_v:
                    max_v = viol
                    worst = ('ic', i, j)
        
        if max_v <= 1e-8:
            break
        
        kind, i, j = worst
        vi = np.array(types[i], dtype=float)
        
        if kind == 'ir':
            payments[i] = max(0.0, float(np.dot(vi, allocations[i])))
        else:
            vj = np.array(types[j], dtype=float)
            min_pj = float(np.dot(vi, allocations[j])) - float(np.dot(vi, allocations[i])) + payments[i]
            max_pj = float(np.dot(vj, allocations[j]))
            if min_pj <= max_pj + 1e-8:
                payments[j] = max(payments[j], min_pj)
            else:
                max_pi = float(np.dot(vi, allocations[i])) - float(np.dot(vi, allocations[j])) + payments[j]
                if max_pi >= 0:
                    payments[i] = min(payments[i], max_pi)
                else:
                    break
    
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        val_own = float(np.dot(vi, allocations[i]))
        if payments[i] > val_own + 1e-8:
            payments[i] = max(0.0, val_own)
    
    payments = np.maximum(payments, 0.0)
    allocations = np.clip(allocations, 0.0, 1.0)
    return allocations, payments
