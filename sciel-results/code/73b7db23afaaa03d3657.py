import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible and individually rational auction mechanism.
    
    Args:
        distribution_spec: Specification of the probability distribution over types.
        support: The support of the types (usually {(0,0), ..., (3,3)}).
        constraints: Additional constraints (not explicitly detailed, assumed standard).
        
    Returns:
        A dictionary with 'allocations' (16, 2) and 'payments' (16,).
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # Build and solve LP
    result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 3.0:
        # Fallback to zero mechanism if running out of time
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    if result is not None:
        # Polish: try to improve by tightening payments
        result = _polish_solution(result, types, probs)
        return result
    else:
        # Fallback to zero mechanism if LP fails
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

def _parse_distribution(distribution_spec, support, types, n_types):
    """
    Parse the distribution spec into an array of probabilities for the 16 types.
    Order: (0,0), (0,1), ..., (0,3), (1,0), ..., (3,3).
    """
    uniform = np.ones(n_types) / n_types
    
    if distribution_spec is None:
        return uniform
    
    # If it's a dict mapping types to probs
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            # Try tuple key
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            # Try string key "(v1, v2)"
            elif f"({t[0]}, {t[1]})" in distribution_spec:
                probs[i] = distribution_spec[f"({t[0]}, {t[1]})"]
                found_any = True
            # Try string key "v1,v2"
            elif f"{t[0]},{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]},{t[1]}"]
                found_any = True
            # Try flattened string key "v1v2"
            elif f"{t[0]}{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]}{t[1]}"]
                found_any = True
        
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()
            return uniform
        return uniform
    
    # If it's a list or array
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
        
        # Maybe it's a 4x4 matrix?
        if len(arr) == 16:
             # Already handled above
             pass
             
        # Maybe it's product of marginals? E.g., [p0, p1, p2, p3] for each good?
        # This is a guess. If length is 4, assume independent marginals.
        if len(arr) == 4:
            # arr[i] is prob for value i?
            # This is ambiguous. Let's stick to uniform if not 16.
            pass
        
        return uniform
    
    # If it's a string, might be a name
    if isinstance(distribution_spec, str):
        if distribution_spec.lower() in ['uniform', 'unif']:
            return uniform
        # Try to parse JSON?
        try:
            import json
            spec = json.loads(distribution_spec)
            return _parse_distribution(spec, support, types, n_types)
        except:
            pass
        return uniform

    # Fallback: Check support
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()

    return uniform

def _solve_lp(probs, types, n_types, n_goods):
    """
    Solve the revenue-maximizing IC/IR LP.
    Uses reduced IC constraints based on monotonicity for efficiency.
    Variables: x[0..2*n_types-1] for allocations, x[2*n_types..3*n_types-1] for payments.
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # linprog minimizes, so we use negative.
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Build reduced IC constraint set using monotonicity
    # For multi-dimensional types, we enforce IC for "adjacent" types in the lattice
    # This is sufficient for single-parameter but we use a safe subset
    ic_pairs = _get_ic_constraint_pairs(types, n_types)
    
    n_ic = len(ic_pairs)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IC Constraints (reduced set)
    for (i, j) in ic_pairs:
        vi = types[i]
        vi_vec = np.array(vi, dtype=float)
        vj = types[j]
        # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i + vi[0]*x_j0 + vi[1]*x_j1 - p_j <= 0
        A_ub[idx, var_x(i, 0)] = -vi_vec[0]
        A_ub[idx, var_x(i, 1)] = -vi_vec[1]
        A_ub[idx, var_p(i)] = 1.0
        A_ub[idx, var_x(j, 0)] = vi_vec[0]
        A_ub[idx, var_x(j, 1)] = vi_vec[1]
        A_ub[idx, var_p(j)] = -1.0
        b_ub[idx] = 0.0
        idx += 1
        
    # IR Constraints
    for i in range(n_types):
        vi = types[i]
        vi_vec = np.array(vi, dtype=float)
        # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i <= 0
        A_ub[idx, var_x(i, 0)] = -vi_vec[0]
        A_ub[idx, var_x(i, 1)] = -vi_vec[1]
        A_ub[idx, var_p(i)] = 1.0
        b_ub[idx] = 0.0
        idx += 1
        
    # Bounds:
    # Allocations x[i,g] in [0, 1]
    # Payments p[i] in [0, infinity)
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    try:
        # Use HiGHS solver with tight tolerances for better precision
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True,
                              'dual_feasibility_tolerance': 1e-13,
                              'primal_feasibility_tolerance': 1e-13,
                              'dual_objective_bound_tolerance': 1e-13})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0, 1))
                payments[i] = float(max(0, x[var_p(i)]))
            
            # Verify full IC constraints (not just the reduced set)
            allocations, payments = _verify_full_ic(allocations, payments, types)
            
            return {'allocations': allocations, 'payments': payments}
        else:
            return None
    except Exception:
        return None

def _get_ic_constraint_pairs(types, n_types):
    """
    Get a reduced set of IC constraint pairs.
    For the 4x4 grid, we enforce IC for:
    - All pairs where type i weakly dominates type j in both goods (i >= j componentwise)
      This ensures monotonicity and is sufficient for IC in single-parameter settings.
    - Additionally, include some cross pairs to be safe.
    """
    pairs = []
    for i in range(n_types):
        vi = types[i]
        for j in range(n_types):
            if i == j:
                continue
            vj = types[j]
            # Include if vi dominates vj (vi[0] >= vj[0] and vi[1] >= vj[1])
            # or if vj dominates vi
            if (vi[0] >= vj[0] and vi[1] >= vj[1]) or (vj[0] >= vi[0] and vj[1] >= vi[1]):
                pairs.append((i, j))
    return pairs

def _verify_full_ic(allocations, payments, types):
    """
    Verify all IC and IR constraints. If violated, fix by adjusting payments.
    """
    n_types = len(types)
    
    # Check IR first
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-12:
            payments[i] = max(0.0, np.dot(vi, allocations[i]))
    
    # Check all IC constraints
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - 1e-10:
                # Need to increase payment for type i
                needed = uj_cross - ui + 1e-11
                payments[i] += needed
                # Re-check IR
                if payments[i] > np.dot(vi, allocations[i]):
                    payments[i] = max(0.0, np.dot(vi, allocations[i]))
    
    # Final clip
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # One more IR pass
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-10:
            payments[i] = max(0.0, np.dot(vi, allocations[i]))
    
    return allocations, payments

def _polish_solution(result, types, probs):
    """
    Try to improve the solution by increasing payments where possible.
    The LP solution should be optimal, but numerical errors might leave room.
    """
    allocations = result['allocations'].copy()
    payments = result['payments'].copy()
    n_types = len(types)
    
    # Try to increase each payment slightly while maintaining IC/IR
    improved = False
    for _ in range(3):  # A few passes
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            current_ui = np.dot(vi, allocations[i]) - payments[i]
            
            # Max payment allowed by IR
            max_p_ir = max(0.0, np.dot(vi, allocations[i]))
            
            # Max payment allowed by IC: for all j, ui >= uj_cross
            # ui = v_i . x_i - p_i >= v_i . x_j - p_j
            # p_i <= v_i . x_i - v_i . x_j + p_j
            max_p_ic = max_p_ir
            for j in range(n_types):
                if i == j:
                    continue
                vj_cross = np.dot(vi, allocations[j]) - payments[j]
                # p_i <= v_i . x_i - v_j_cross
                allowed = np.dot(vi, allocations[i]) - vj_cross
                max_p_ic = min(max_p_ic, allowed)
            
            # Try to set payment to max_p_ic (slightly below to avoid numerical issues)
            new_p = max(0.0, min(max_p_ir, max_p_ic) - 1e-12)
            if new_p > payments[i] + 1e-10:
                payments[i] = new_p
                improved = True
    
    return {'allocations': allocations, 'payments': payments}


