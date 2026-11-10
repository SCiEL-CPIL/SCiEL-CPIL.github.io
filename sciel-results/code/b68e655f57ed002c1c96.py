"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _parse_distribution(distribution_spec):
    """Parse distribution_spec into a 16-element probability vector."""
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    
    if isinstance(distribution_spec, dict):
        for key in ['type_probs', 'probs', 'probabilities', 'p']:
            if key in distribution_spec:
                arr = np.array(distribution_spec[key], dtype=float)
                if len(arr) == 16:
                    arr = arr / arr.sum()
                    return arr
        for k, v in distribution_spec.items():
            if isinstance(v, (list, np.ndarray)) and len(v) == 16:
                arr = np.array(v, dtype=float)
                arr = arr / arr.sum()
                return arr
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == 16:
            arr = np.array(distribution_spec, dtype=float)
            arr = arr / arr.sum()
            return arr
    
    return np.ones(16) / 16.0

def _solve_lp_auction(probs, types, deadline=None):
    """
    Solve the LP for optimal incentive-compatible auction.
    
    Variables:
    - q[i, j] for i in 0..15, j in 0..1: allocation probability for good j to type i
    - p[i] for i in 0..15: payment to type i
    
    Objective: maximize sum_i probs[i] * p[i]
    """
    n_types = 16
    n_goods = 2
    
    n_vars = 32 + 16  # 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # 1. Feasibility: -q[i,j] <= 0 (q >= 0)
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(n_vars)
            row[i * 2 + j] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # 2. q[i,j] <= 1
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(n_vars)
            row[i * 2 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    # 3. Nonnegative payments: -p[i] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # 4. Individual rationality: p_i - v_i . q_i <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[32 + i] = 1.0
        row[i * 2 + 0] = -v1
        row[i * 2 + 1] = -v2
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # 5. IC: v_i . (q_i - q_j) - p_i + p_j <= 0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i * 2 + 0] = v1_i
            row[i * 2 + 1] = v2_i
            row[j * 2 + 0] = -v1_i
            row[j * 2 + 1] = -v2_i
            row[32 + i] = -1.0
            row[32 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'time_limit': max(0.1, (deadline - time.time()) if deadline else 30.0)})
    
    if not result.success:
        return None, None
    
    x = result.x
    q = x[:32].reshape(16, 2)
    p = x[32:48]
    
    return q, p

def _solve_lp_utility(probs, types, deadline=None):
    """
    Alternative LP formulation parameterized by expected utility U(v1, v2).
    
    For a single additive bidder, IC implies the allocation is the gradient of U.
    We parameterize U directly and derive allocations/payments.
    
    For type i with values (v1_i, v2_i):
    - Allocation q_i = (∂U/∂v1, ∂U/∂v2) at type i
    - Payment p_i = v1_i * q_i1 + v2_i * q_i2 - U_i
    
    For discrete support {0,1,2,3}, we use finite differences:
    ∂U/∂v1 at v1 = (U(v1+1, v2) - U(v1, v2)) / 1 for v1 < 3
    ∂U/∂v2 at v2 = (U(v1, v2+1) - U(v1, v2)) / 1 for v2 < 3
    
    This is a linear formulation in U variables.
    """
    n_types = 16
    n_goods = 2
    
    # Variables: U[i] for i in 0..15 (16 vars)
    n_vars = 16
    
    # Objective: maximize sum_i probs[i] * p_i
    # p_i = v1_i * q_i1 + v2_i * q_i2 - U_i
    # q_i1 = U(v1_i+1, v2_i) - U(v1_i, v2_i) if v1_i < 3, else 0 (or use backward diff)
    # q_i2 = U(v1_i, v2_i+1) - U(v1_i, v2_i) if v2_i < 3, else 0
    
    # For boundary types (v1=3 or v2=3), use backward difference
    # q_i1 at v1=3: U(3, v2) - U(2, v2)
    # q_i2 at v2=3: U(v1, 3) - U(v1, 2)
    
    def get_type_idx(v1, v2):
        return v1 * 4 + v2
    
    # Build objective: max sum_i probs[i] * (v1_i * q_i1 + v2_i * q_i2 - U_i)
    c = np.zeros(n_vars)
    
    for i in range(n_types):
        v1, v2 = types[i]
        
        # q_i1
        if v1 < 3:
            # q_i1 = U(v1+1, v2) - U(v1, v2)
            idx_forward = get_type_idx(v1 + 1, v2)
            idx_current = i
            c[idx_forward] += probs[i] * v1
            c[idx_current] -= probs[i] * v1
        else:
            # q_i1 = U(3, v2) - U(2, v2)
            idx_current = i
            idx_backward = get_type_idx(2, v2)
            c[idx_current] += probs[i] * v1
            c[idx_backward] -= probs[i] * v1
        
        # q_i2
        if v2 < 3:
            # q_i2 = U(v1, v2+1) - U(v1, v2)
            idx_forward = get_type_idx(v1, v2 + 1)
            idx_current = i
            c[idx_forward] += probs[i] * v2
            c[idx_current] -= probs[i] * v2
        else:
            # q_i2 = U(v1, 3) - U(v1, 2)
            idx_current = i
            idx_backward = get_type_idx(v1, 2)
            c[idx_current] += probs[i] * v2
            c[idx_backward] -= probs[i] * v2
        
        # -U_i term
        c[i] -= probs[i]
    
    # Constraints:
    # 1. Allocation feasibility: 0 <= q_ij <= 1
    # 2. Nonnegative payments: p_i >= 0
    # 3. IR: U_i >= 0 (since U_i = v_i . q_i - p_i, and IR says v_i . q_i - p_i >= 0)
    # 4. Monotonicity of U (implied by q >= 0)
    
    A_ub_rows = []
    b_ub_vals = []
    
    # For each type i, compute q_i1 and q_i2 in terms of U
    for i in range(n_types):
        v1, v2 = types[i]
        
        # q_i1 >= 0
        row = np.zeros(n_vars)
        if v1 < 3:
            idx_forward = get_type_idx(v1 + 1, v2)
            row[idx_forward] = 1.0
            row[i] = -1.0
        else:
            row[i] = 1.0
            row[get_type_idx(2, v2)] = -1.0
        A_ub_rows.append(-row)  # -q_i1 <= 0
        b_ub_vals.append(0.0)
        
        # q_i1 <= 1
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
        # q_i2 >= 0
        row = np.zeros(n_vars)
        if v2 < 3:
            idx_forward = get_type_idx(v1, v2 + 1)
            row[idx_forward] = 1.0
            row[i] = -1.0
        else:
            row[i] = 1.0
            row[get_type_idx(v1, 2)] = -1.0
        A_ub_rows.append(-row)
        b_ub_vals.append(0.0)
        
        # q_i2 <= 1
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
        # p_i >= 0: v1*q_i1 + v2*q_i2 - U_i >= 0 => U_i - v1*q_i1 - v2*q_i2 <= 0
        row = np.zeros(n_vars)
        row[i] = 1.0
        
        # -v1 * q_i1
        if v1 < 3:
            row[get_type_idx(v1 + 1, v2)] -= v1
            row[i] += v1
        else:
            row[i] += v1
            row[get_type_idx(2, v2)] -= v1
        
        # -v2 * q_i2
        if v2 < 3:
            row[get_type_idx(v1, v2 + 1)] -= v2
            row[i] += v2
        else:
            row[i] += v2
            row[get_type_idx(v1, 2)] -= v2
        
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # U_i >= 0: -U_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = [(0, None)] * n_vars
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'time_limit': max(0.1, (deadline - time.time()) if deadline else 30.0)})
    
    if not result.success:
        return None, None
    
    U = result.x
    
    # Derive allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(n_types):
        v1, v2 = types[i]
        
        # q_i1
        if v1 < 3:
            q1 = U[get_type_idx(v1 + 1, v2)] - U[i]
        else:
            q1 = U[i] - U[get_type_idx(2, v2)]
        
        # q_i2
        if v2 < 3:
            q2 = U[get_type_idx(v1, v2 + 1)] - U[i]
        else:
            q2 = U[i] - U[get_type_idx(v1, 2)]
        
        allocations[i, 0] = q1
        allocations[i, 1] = q2
        payments[i] = v1 * q1 + v2 * q2 - U[i]
    
    return allocations, payments

def _verify_and_repair(allocations, payments, probs, types, tol=1e-8):
    """Verify constraints and repair if needed. Returns (valid, allocations, payments, revenue)."""
    n_types = 16
    
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        allocations = np.clip(allocations, 0, 1)
    
    if np.any(payments < -tol):
        payments = np.maximum(payments, 0)
    
    for i in range(n_types):
        v1, v2 = types[i]
        util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        if util < -tol:
            payments[i] = max(0, v1 * allocations[i, 0] + v2 * allocations[i, 1])
    
    max_violation = 0.0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        true_util = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
        for j in range(n_types):
            misreport_util = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
            violation = misreport_util - true_util
            if violation > max_violation:
                max_violation = violation
    
    if max_violation > tol:
        return False, allocations, payments, 0.0
    
    revenue = np.sum(probs * payments)
    return True, allocations, payments, revenue

def _fallback_result(probs, types):
    """Return a simple valid fallback auction."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(16):
        v1, v2 = types[i]
        allocations[i, 0] = 1.0 if v1 >= 1 else 0.0
        allocations[i, 1] = 1.0 if v2 >= 1 else 0.0
        payments[i] = (1.0 if v1 >= 1 else 0.0) + (1.0 if v2 >= 1 else 0.0)
    
    return allocations, payments

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, three fixed distributions.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    t0 = time.time()
    DEADLINE = 85.0
    
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    
    probs = _parse_distribution(distribution_spec)
    
    # Start with Myerson-style independent approach
    p1 = np.zeros(4)
    p2 = np.zeros(4)
    for i in range(16):
        v1, v2 = types[i]
        p1[v1] += probs[i]
        p2[v2] += probs[i]
    
    def compute_allocation(p):
        q = np.zeros(4)
        for v in range(4):
            if v == 3:
                coeff = 3 * p[3]
            else:
                coeff = v * p[v] - np.sum(p[v+1:4])
            q[v] = 1.0 if coeff > 1e-12 else 0.0
        return q
    
    q1_myerson = compute_allocation(p1)
    q2_myerson = compute_allocation(p2)
    
    allocations_myerson = np.zeros((16, 2))
    payments_myerson = np.zeros(16)
    for i in range(16):
        v1, v2 = types[i]
        allocations_myerson[i, 0] = q1_myerson[v1]
        allocations_myerson[i, 1] = q2_myerson[v2]
        pay1 = v1 * q1_myerson[v1] - np.sum(q1_myerson[:v1]) if v1 > 0 else 0.0
        pay2 = v2 * q2_myerson[v2] - np.sum(q2_myerson[:v2]) if v2 > 0 else 0.0
        payments_myerson[i] = pay1 + pay2
    
    payments_myerson = np.maximum(payments_myerson, 0.0)
    valid_myerson, alloc_m, pay_m, rev_m = _verify_and_repair(allocations_myerson.copy(), payments_myerson.copy(), probs, types)
    
    best_alloc = alloc_m.copy()
    best_pay = pay_m.copy()
    best_rev = rev_m if valid_myerson else 0.0
    
    # Try utility-parameterized LP
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 10:
        try:
            alloc_u, pay_u = _solve_lp_utility(probs, types, deadline=DEADLINE - 5)
            if alloc_u is not None:
                valid_u, alloc_u, pay_u, rev_u = _verify_and_repair(alloc_u.copy(), pay_u.copy(), probs, types)
                if valid_u and rev_u > best_rev:
                    best_alloc = alloc_u.copy()
                    best_pay = pay_u.copy()
                    best_rev = rev_u
        except Exception:
            pass
    
    # Try standard LP
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 10:
        try:
            q_lp, p_lp = _solve_lp_auction(probs, types, deadline=DEADLINE - 5)
            if q_lp is not None:
                valid_lp, alloc_lp, pay_lp, rev_lp = _verify_and_repair(q_lp.copy(), p_lp.copy(), probs, types)
                if valid_lp and rev_lp > best_rev:
                    best_alloc = alloc_lp.copy()
                    best_pay = pay_lp.copy()
                    best_rev = rev_lp
        except Exception:
            pass
    
    # Final verification
    valid_final, alloc_final, pay_final, rev_final = _verify_and_repair(best_alloc.copy(), best_pay.copy(), probs, types)
    
    if not valid_final:
        alloc_final, pay_final = _fallback_result(probs, types)
        valid_final, alloc_final, pay_final, rev_final = _verify_and_repair(alloc_final, pay_final, probs, types)
    
    return {'allocations': alloc_final, 'payments': pay_final}
# EVOLVE-BLOCK-END
