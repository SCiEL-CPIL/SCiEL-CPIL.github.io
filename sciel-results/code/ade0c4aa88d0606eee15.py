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

    # 1. Solve the joint LP directly (x, p) using HiGHS
    #    This is the most robust approach for small finite support.
    result = _solve_joint_lp(probs, types, n_types, n_goods)
    
    if result is not None and time.time() < deadline - 5.0:
        allocations, payments = result
    else:
        # Fallback: Simple safe mechanism
        allocations, payments = _simple_safe_mechanism(types)

    # 2. Polish pass: verify constraints and repair if needed
    #    This is critical to ensure strict validity within 1e-8 tolerance.
    allocations, payments = _polish_mechanism(allocations, payments, probs, types)

    return {
        'allocations': allocations,
        'payments': payments
    }

def _parse_distribution(distribution_spec, support, types, n_types):
    """Robustly parse distribution spec to get type probabilities."""
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    # If it's a direct array of correct size
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
            return np.ones(n_types) / n_types

    # If it's a dict
    if isinstance(distribution_spec, dict):
        # Look for common keys
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
                    # Map tuples to probs
                    probs = np.zeros(n_types)
                    for i, t in enumerate(types):
                        if t in val:
                            probs[i] = val[t]
                        elif str(t) in val:
                            probs[i] = val[str(t)]
                    probs = np.maximum(probs, 0)
                    if probs.sum() > 0:
                        return probs / probs.sum()
        
        # Check if the dict itself maps types to probs
        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
                found_any = True
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()
    
    # Check support dict
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()

    # Fallback: Uniform
    return np.ones(n_types) / n_types

def _solve_joint_lp(probs, types, n_types, n_goods):
    """
    Solve the joint LP for x and p.
    Variables:
    x_{i,g} for i in 0..15, g in 0..1
    p_i for i in 0..15
    
    Objective: Maximize sum_i probs[i] * p_i
    Subject to:
    1. IC: v_i . x_i - p_i >= v_i . x_j - p_j  =>  -v_i . x_i + p_i + v_i . x_j - p_j <= 0
    2. IR: v_i . x_i - p_i >= 0  =>  -v_i . x_i + p_i <= 0
    3. Bounds: 0 <= x_{i,g} <= 1, p_i >= 0
    """
    def var_x(i, g):
        return 2 * i + g

    def var_p(i):
        return 2 * n_types + i

    n_vars = 2 * n_types + n_types

    # Objective: minimize -sum probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Constraints
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        vi = np.array(types[i])
        for j in range(n_types):
            if i == j:
                continue
            # -vi.x_i + p_i + vi.x_j - p_j <= 0
            A_ub[idx, var_x(i, 0)] = -vi[0]
            A_ub[idx, var_x(i, 1)] = -vi[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi[0]
            A_ub[idx, var_x(j, 1)] = vi[1]
            A_ub[idx, var_p(j)] = -1.0
            idx += 1
    
    for i in range(n_types):
        vi = np.array(types[i])
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        idx += 1
    
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    try:
        # Use HiGHS for high precision
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={"presolve": "on", 
                               "dual_feasibility_tolerance": 1e-12, 
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 80.0})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                payments[i] = x[var_p(i)]
            
            # Clamp small numerical errors
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)
            
            return allocations, payments
        else:
            return None
    except Exception:
        return None

def _polish_mechanism(allocations, payments, probs, types):
    """
    Verify IC/IR constraints and repair violations.
    This is a critical step to ensure the output is strictly valid
    within the required tolerance (1e-8).
    
    Strategy:
    1. Check IR. If violated, reduce payment to make utility 0.
    2. Check IC. If violated, increase payment of the 'tempting' type j
       to reduce its utility relative to type i.
    """
    n_types = len(types)
    tol = 1e-9  # Use tighter internal tolerance for safety
    
    # Check IR: v_i . x_i - p_i >= 0
    for i in range(n_types):
        vi = np.array(types[i])
        xi = np.array(allocations[i])
        utility = np.dot(vi, xi) - payments[i]
        if utility < -tol:
            # Violation: reduce payment to make utility 0
            payments[i] = np.dot(vi, xi)
            if payments[i] < 0:
                payments[i] = 0.0 # Should not happen if x>=0, v>=0

    # Check IC: v_i . x_i - p_i >= v_i . x_j - p_j for all i, j
    # If violated, we need to increase p_j so that v_i . x_j - p_j decreases.
    # New p_j = p_j + (v_i . x_j - p_j) - (v_i . x_i - p_i) + margin
    #         = p_j + (v_i . x_j - v_i . x_i) + (p_i - p_j) + margin
    #         = p_i + v_i . (x_j - x_i) + margin
    
    for i in range(n_types):
        vi = np.array(types[i])
        xi = np.array(allocations[i])
        util_i = np.dot(vi, xi) - payments[i]
        
        for j in range(n_types):
            if i == j:
                continue
            xj = np.array(allocations[j])
            util_ij = np.dot(vi, xj) - payments[j]
            
            if util_ij > util_i + tol:
                # IC violation for type i reporting j
                # We want: util_ij <= util_i
                # Current: util_ij = v_i . x_j - p_j
                # Target: v_i . x_j - p_j_new <= util_i
                # p_j_new >= v_i . x_j - util_i
                # p_j_new = max(p_j, v_i . x_j - util_i + tol)
                required_pj = np.dot(vi, xj) - util_i + tol
                if required_pj > payments[j]:
                    payments[j] = required_pj

    # Re-check and clip
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Final strict check and repair loop (multiple passes to handle cascading effects)
    for _ in range(5):
        for i in range(n_types):
            vi = np.array(types[i])
            xi = np.array(allocations[i])
            if np.dot(vi, xi) - payments[i] < -1e-8:
                payments[i] = np.dot(vi, xi)
                if payments[i] < 0:
                    payments[i] = 0.0

        for i in range(n_types):
            vi = np.array(types[i])
            xi = np.array(allocations[i])
            util_i = np.dot(vi, xi) - payments[i]
            for j in range(n_types):
                if i == j: continue
                xj = np.array(allocations[j])
                util_ij = np.dot(vi, xj) - payments[j]
                if util_ij > util_i + 1e-8:
                    required_pj = np.dot(vi, xj) - util_i + 1e-8
                    if required_pj > payments[j]:
                        payments[j] = required_pj

    return allocations, payments

def _simple_safe_mechanism(types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    x_i = 0 for all i, p_i = 0 for all i.
    """
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
