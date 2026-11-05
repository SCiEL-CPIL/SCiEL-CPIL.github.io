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
    
    # 1. Solve full LP to get candidate allocation x
    x_alloc, p_alloc = _solve_full_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 10.0:
        # Fallback if full LP is too slow
        return _simple_safe_mechanism(types)
    
    # 2. Try ironing-based approach to potentially improve allocation
    x_iron = _solve_ironing_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 8.0:
        # Use whichever allocation we have
        best_alloc, best_pay = _refine_payments(x_alloc, probs, types)
    else:
        # Compare revenue from both approaches and pick the better one
        best_alloc, best_pay = _compare_and_select(x_alloc, p_alloc, x_iron, probs, types)
    
    # 3. Final safety check and clipping
    best_alloc = np.clip(best_alloc, 0.0, 1.0)
    best_pay = np.maximum(best_pay, 0.0)
    
    # 4. Verification and fix pass
    best_alloc, best_pay = _verify_and_fix(best_alloc, best_pay, probs, types)
    
    # 5. Polish pass with tighter tolerances if time permits
    if time.time() < deadline - 5.0:
        best_alloc, best_pay = _polish_solution(best_alloc, best_pay, probs, types)
    
    # Final safety
    best_alloc = np.clip(best_alloc, 0.0, 1.0)
    best_pay = np.maximum(best_pay, 0.0)
    
    return {
        'allocations': best_alloc,
        'payments': best_pay
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

def _solve_full_lp(probs, types, n_types, n_goods):
    """
    Solve the full LP to find optimal allocation x and payment p.
    Returns (allocations, payments).
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types
    
    # Objective: maximize sum_i probs[i] * p[i] -> minimize -sum
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Constraints:
    # 1. IC: p_i - p_j <= v_i . x_i - v_i . x_j  => -v_i . x_i + p_i + v_i . x_j - p_j <= 0
    # 2. IR: -v_i . x_i + p_i <= 0
    # 3. Bounds: 0 <= x <= 1, p >= 0
    
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
            vj = np.array(types[j])
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
        # Use HiGHS for speed and robustness
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                payments[i] = x[var_p(i)]
            return allocations, payments
        else:
            # Fallback: Zero allocation
            return np.zeros((n_types, n_goods)), np.zeros(n_types)
    except Exception:
        return np.zeros((n_types, n_goods)), np.zeros(n_types)

def _solve_ironing_lp(probs, types, n_types, n_goods):
    """
    Solve the ironing-based LP. In multi-dimensional auction design, the optimal
    allocation can be found by solving a convex program. For finite types, we can
    formulate this as an LP over the allocation variables with IC constraints
    expressed via the envelope theorem.
    
    The key insight: for a given allocation rule x(v), the IC constraints are
    equivalent to requiring that the utility function U(v) = v.x(v) - p(v) is
    convex in v. For finite types, this means:
    U_i >= U_j + v_i . (x_j - x_i) for all i, j (tangent plane condition)
    
    We maximize expected revenue = sum_i probs[i] * (v_i . x_i - U_i)
    subject to:
    - 0 <= x_i <= 1
    - U_i >= 0 (IR)
    - U_i >= U_j + v_i . x_j - v_i . x_i for all i, j (IC via convexity)
    """
    n_vars = 2 * n_types + n_types  # x_i0, x_i1, U_i
    
    def var_x(i, g):
        return 2 * i + g
    
    def var_U(i):
        return 2 * n_types + i
    
    # Objective: maximize sum_i probs[i] * (v_i . x_i - U_i)
    # = maximize sum_i probs[i] * v_i[0] * x_i0 + sum_i probs[i] * v_i[1] * x_i1 - sum_i probs[i] * U_i
    c = np.zeros(n_vars)
    for i in range(n_types):
        vi = np.array(types[i])
        c[var_x(i, 0)] = -probs[i] * vi[0]
        c[var_x(i, 1)] = -probs[i] * vi[1]
        c[var_U(i)] = probs[i]  # We're minimizing, so -(-probs[i]*U_i) = probs[i]*U_i
    
    # Constraints:
    # 1. IR: -U_i <= 0
    # 2. IC (convexity): U_i - U_j - v_i . x_j + v_i . x_i <= 0 for all i != j
    #    i.e., U_i + v_i . x_i <= U_j + v_i . x_j
    #    Rearranged: U_i - U_j + v_i . x_i - v_i . x_j <= 0
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IR: -U_i <= 0
    for i in range(n_types):
        A_ub[idx, var_U(i)] = -1.0
        b_ub[idx] = 0.0
        idx += 1
    
    # IC: U_i - U_j + v_i . x_i - v_i . x_j <= 0
    for i in range(n_types):
        vi = np.array(types[i])
        for j in range(n_types):
            if i == j:
                continue
            # U_i - U_j + vi[0]*x_i0 - vi[0]*x_j0 + vi[1]*x_i1 - vi[1]*x_j1 <= 0
            A_ub[idx, var_U(i)] = 1.0
            A_ub[idx, var_U(j)] = -1.0
            A_ub[idx, var_x(i, 0)] = vi[0]
            A_ub[idx, var_x(i, 1)] = vi[1]
            A_ub[idx, var_x(j, 0)] = -vi[0]
            A_ub[idx, var_x(j, 1)] = -vi[1]
            idx += 1
    
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            U = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                U[i] = x[var_U(i)]
            
            # Compute payments: p_i = v_i . x_i - U_i
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i])
                xi = np.array(allocations[i])
                payments[i] = np.dot(vi, xi) - U[i]
                payments[i] = max(payments[i], 0.0)
            
            return allocations
        else:
            return np.zeros((n_types, n_goods))
    except Exception:
        return np.zeros((n_types, n_goods))

def _refine_payments(allocations, probs, types):
    """
    Given an allocation x, solve for the optimal payments p that satisfy IC and IR
    and maximize revenue. This is equivalent to minimizing total expected utility.
    
    Variables: U_i (utility of type i)
    Minimize: sum_i probs[i] * U_i
    Subject to:
    1. U_i >= 0  (IR)
    2. U_i - U_j >= v_i . x_j - v_j . x_j  (IC) for all i, j
    """
    n_types = len(types)
    
    # Objective: minimize sum probs[i] * U_i
    c = probs.copy()
    
    # Constraints:
    # 1. -U_i <= 0  (U_i >= 0)
    # 2. -U_i + U_j <= v_j . x_j - v_i . x_j  (Rearranged from U_i - U_j >= v_i . x_j - v_j . x_j)
    
    n_vars = n_types
    n_ineq = n_types + n_types * (n_types - 1)
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IR: -U_i <= 0
    for i in range(n_types):
        A_ub[idx, i] = -1.0
        b_ub[idx] = 0.0
        idx += 1
    
    # IC: -U_i + U_j <= v_j . x_j - v_i . x_j
    for i in range(n_types):
        vi = np.array(types[i])
        
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j])
            xj = np.array(allocations[j])
            # RHS: v_j . x_j - v_i . x_j
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1
            
    bounds = [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
        
        if res.success:
            U = res.x
            # Ensure U >= 0 (should be enforced by bounds, but clip for safety)
            U = np.maximum(U, 0.0)
            
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i])
                xi = np.array(allocations[i])
                p_i = np.dot(vi, xi) - U[i]
                # Payment must be non-negative (IR ensures U_i <= v_i . x_i, so p_i >= 0)
                # But due to numerical errors, U might be slightly > v.i . x.i
                if p_i < 0:
                    p_i = 0.0
                payments[i] = p_i
            
            return allocations, payments
        else:
            return _simple_safe_mechanism(types)
            
    except Exception:
        return _simple_safe_mechanism(types)

def _compare_and_select(alloc1, pay1, alloc2, probs, types):
    """
    Compare two allocation/payment pairs and return the one with higher expected revenue,
    after verifying IC/IR constraints.
    """
    n_types = len(types)
    
    def expected_revenue(alloc, pay):
        return np.sum(probs * pay)
    
    def is_valid(alloc, pay):
        # Check IR: v_i . x_i - p_i >= 0
        for i in range(n_types):
            vi = np.array(types[i])
            xi = np.array(alloc[i])
            if np.dot(vi, xi) - pay[i] < -1e-8:
                return False
        # Check IC: v_i . x_i - p_i >= v_i . x_j - p_j for all i, j
        for i in range(n_types):
            vi = np.array(types[i])
            xi = np.array(alloc[i])
            ui = np.dot(vi, xi) - pay[i]
            for j in range(n_types):
                xj = np.array(alloc[j])
                uj = np.dot(vi, xj) - pay[j]
                if ui - uj < -1e-8:
                    return False
        # Check bounds
        if np.any(alloc < -1e-8) or np.any(alloc > 1 + 1e-8):
            return False
        if np.any(pay < -1e-8):
            return False
        return True
    
    valid1 = is_valid(alloc1, pay1)
    valid2 = is_valid(alloc2, pay1)  # Need to compute pay for alloc2
    
    # Compute payments for alloc2
    alloc2_clean, pay2 = _refine_payments(alloc2, probs, types)
    valid2 = is_valid(alloc2_clean, pay2)
    
    if valid1 and valid2:
        rev1 = expected_revenue(alloc1, pay1)
        rev2 = expected_revenue(alloc2_clean, pay2)
        if rev2 > rev1:
            return alloc2_clean, pay2
        else:
            return alloc1, pay1
    elif valid1:
        return alloc1, pay1
    elif valid2:
        return alloc2_clean, pay2
    else:
        return _simple_safe_mechanism(types)

def _verify_and_fix(allocations, payments, probs, types):
    """
    Verify IC/IR constraints and fix any violations by adjusting payments.
    """
    n_types = len(types)
    
    # Check and fix IR: ensure v_i . x_i - p_i >= 0
    for i in range(n_types):
        vi = np.array(types[i])
        xi = np.array(allocations[i])
        max_pay = np.dot(vi, xi)
        if payments[i] > max_pay + 1e-10:
            payments[i] = max(0.0, max_pay)
    
    # Check IC and fix by increasing payments if needed
    # IC: v_i . x_i - p_i >= v_i . x_j - p_j for all i, j
    # => p_i <= v_i . x_i - v_i . x_j + p_j
    # To fix, we may need to increase some payments or decrease others
    # Simple approach: re-solve the payment LP with the current allocation
    if time.time() < 85.0:
        try:
            alloc_clean, pay_clean = _refine_payments(allocations, probs, types)
            return alloc_clean, pay_clean
        except:
            pass
    
    # Fallback: just clip
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    return allocations, payments

def _polish_solution(allocations, payments, probs, types):
    """
    Final polish pass: re-solve with tighter tolerances to get a more precise solution.
    """
    try:
        # Re-solve the payment refinement with tighter tolerances
        n_types = len(types)
        c = probs.copy()
        n_vars = n_types
        n_ineq = n_types + n_types * (n_types - 1)
        
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        
        idx = 0
        for i in range(n_types):
            A_ub[idx, i] = -1.0
            b_ub[idx] = 0.0
            idx += 1
        
        for i in range(n_types):
            vi = np.array(types[i])
            for j in range(n_types):
                if i == j:
                    continue
                vj = np.array(types[j])
                xj = np.array(allocations[j])
                rhs = np.dot(vj, xj) - np.dot(vi, xj)
                A_ub[idx, i] = -1.0
                A_ub[idx, j] = 1.0
                b_ub[idx] = rhs
                idx += 1
        
        bounds = [(0, None)] * n_types
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12, 
                               "primal_feasibility_tolerance": 1e-12,
                               "dual_edge_tolerance": 1e-12, "primal_edge_tolerance": 1e-12})
        
        if res.success:
            U = np.maximum(res.x, 0.0)
            payments_new = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i])
                xi = np.array(allocations[i])
                p_i = np.dot(vi, xi) - U[i]
                payments_new[i] = max(p_i, 0.0)
            
            # Verify the polished solution is valid
            if _is_valid_mechanism(allocations, payments_new, types):
                return allocations, payments_new
            else:
                return allocations, payments
        else:
            return allocations, payments
    except Exception:
        return allocations, payments

def _is_valid_mechanism(allocations, payments, types):
    """Check if a mechanism is IC, IR, and has valid bounds."""
    n_types = len(types)
    
    # Check bounds
    if np.any(allocations < -1e-8) or np.any(allocations > 1 + 1e-8):
        return False
    if np.any(payments < -1e-8):
        return False
    
    # Check IR
    for i in range(n_types):
        vi = np.array(types[i])
        xi = np.array(allocations[i])
        if np.dot(vi, xi) - payments[i] < -1e-8:
            return False
    
    # Check IC
    for i in range(n_types):
        vi = np.array(types[i])
        xi = np.array(allocations[i])
        ui = np.dot(vi, xi) - payments[i]
        for j in range(n_types):
            xj = np.array(allocations[j])
            uj = np.dot(vi, xj) - payments[j]
            if ui - uj < -1e-8:
                return False
    
    return True

def _simple_safe_mechanism(types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    x_i = 0 for all i, p_i = 0 for all i.
    """
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
