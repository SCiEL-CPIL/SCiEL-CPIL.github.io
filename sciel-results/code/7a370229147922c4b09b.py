"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # Build and solve LP
    result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 3.0:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    if result is not None:
        return result
    else:
        # Fallback: safe mechanism with zero allocations
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

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
    
    # IC: for all i != j: v_i·x_i - p_i >= v_i·x_j - p_j
    # => -v_i·x_i + p_i + v_i·x_j - p_j <= 0
    # IR: for all i: v_i·x_i - p_i >= 0 => -v_i·x_i + p_i <= 0
    
    n_ic = n_types * (n_types - 1)  # only i != j
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
    
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0, 1))
                payments[i] = float(max(0, x[var_p(i)]))
            
            # High-precision polish pass
            allocations, payments = _polish_solution(allocations, payments, types, probs)
            
            # Final verification
            if _is_valid(allocations, payments, types, probs):
                return {'allocations': allocations, 'payments': payments}
            else:
                # Fallback to simpler fix
                allocations, payments = _verify_and_fix(allocations, payments, types, probs)
                if _is_valid(allocations, payments, types, probs):
                    return {'allocations': allocations, 'payments': payments}
                return None
        else:
            return None
    except Exception:
        return None

def _is_valid(allocations, payments, types, probs, tol=1e-8):
    """Check if the mechanism satisfies all constraints."""
    n_types = len(types)
    
    # Check allocations in [0,1]
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    # Check payments non-negative
    if np.any(payments < -tol):
        return False
    
    # Check IR: v_i · x_i - p_i >= 0
    for i in range(n_types):
        vi = types[i]
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -tol:
            return False
    
    # Check IC: v_i · x_i - p_i >= v_i · x_j - p_j for all i, j
    for i in range(n_types):
        vi = types[i]
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - tol:
                return False
    
    return True

def _polish_solution(allocations, payments, types, probs):
    """High-precision polish using iterative projection onto feasible set."""
    n_types = len(types)
    n_goods = allocations.shape[1]
    
    alloc = allocations.copy().astype(float)
    pay = payments.copy().astype(float)
    
    # Clip to bounds
    alloc = np.clip(alloc, 0, 1)
    pay = np.maximum(pay, 0)
    
    # Iterative refinement: project onto IR and IC constraints
    for iteration in range(50):
        changed = False
        
        # Project onto IR: p_i <= v_i · x_i
        for i in range(n_types):
            vi = types[i]
            max_pay = np.dot(vi, alloc[i])
            if pay[i] > max_pay + 1e-15:
                pay[i] = max(max_pay, 0)
                changed = True
        
        # Project onto IC: for each i, p_i <= min_j (v_i·x_i - v_i·x_j + p_j)
        for i in range(n_types):
            vi = types[i]
            base_util = np.dot(vi, alloc[i])
            # p_i <= base_util - (v_i·x_j - p_j) for all j
            # => p_i <= base_util + p_j - v_i·x_j
            min_allowed = base_util  # IR bound
            for j in range(n_types):
                if j == i:
                    continue
                cross_term = np.dot(vi, alloc[j]) - pay[j]
                # Need: base_util - p_i >= cross_term
                # => p_i <= base_util - cross_term
                upper = base_util - cross_term
                if upper < min_allowed:
                    min_allowed = upper
            
            if pay[i] > min_allowed + 1e-15:
                pay[i] = max(min_allowed, 0)
                changed = True
        
        if not changed:
            break
    
    return alloc, pay

def _verify_and_fix(allocations, payments, types, probs):
    """Verify IC/IR and fix small numerical violations."""
    n_types = len(types)
    n_goods = allocations.shape[1]
    
    alloc = allocations.copy().astype(float)
    pay = payments.copy().astype(float)
    
    # Clip to bounds
    alloc = np.clip(alloc, 0, 1)
    pay = np.maximum(pay, 0)
    
    # Fix IR: ensure p_i <= v_i · x_i
    for i in range(n_types):
        vi = types[i]
        max_pay = np.dot(vi, alloc[i])
        if pay[i] > max_pay + 1e-10:
            pay[i] = max(max_pay, 0)
    
    # Fix IC iteratively
    for _ in range(20):
        changed = False
        for i in range(n_types):
            vi = types[i]
            base_util = np.dot(vi, alloc[i])
            min_allowed = base_util
            for j in range(n_types):
                if j == i:
                    continue
                cross_term = np.dot(vi, alloc[j]) - pay[j]
                upper = base_util - cross_term
                if upper < min_allowed:
                    min_allowed = upper
            
            if pay[i] > min_allowed + 1e-10:
                pay[i] = max(min_allowed, 0)
                changed = True
        
        if not changed:
            break
    
    return alloc, pay
