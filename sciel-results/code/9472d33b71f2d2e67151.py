"""Improved program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction with one additive bidder, two goods,
    values in {0,1,2,3}, using the optimal mechanism derived from an LP.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # 1. Define Types and Bundles
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    n_types = 16
    n_bundles = 4
    bundles = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    # 2. Parse Distribution
    probs = np.zeros(16)
    try:
        if isinstance(distribution_spec, dict):
            for i, t in enumerate(types):
                if t in distribution_spec:
                    probs[i] = distribution_spec[t]
                elif str(t) in distribution_spec:
                    probs[i] = distribution_spec[str(t)]
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(16) / 16.0
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == 16:
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = np.ones(16) / 16.0
            else:
                probs = np.ones(16) / 16.0
        else:
            probs = np.ones(16) / 16.0
    except Exception:
        probs = np.ones(16) / 16.0
        
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0

    # 3. Setup Value Matrix
    # val[i, j] = value of bundle j for type i
    val = np.zeros((n_types, n_bundles))
    for i, (v1, v2) in enumerate(types):
        for j, (x1, x2) in enumerate(bundles):
            val[i, j] = v1 * x1 + v2 * x2

    # 4. Define LP Variables
    # x[i, j] for i in 0..15, j in 0..3 (64 vars)
    # p[i] for i in 0..15 (16 vars)
    # Total vars: 80
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # Minimize -sum_i probs[i] * p[i]
    c = np.zeros(80)
    for i in range(n_types):
        c[64 + i] = -probs[i]

    # 5. Equality Constraints
    # Sum_j x[i, j] = 1 for all i
    A_eq = np.zeros((n_types, 80))
    b_eq = np.ones(n_types)
    for i in range(n_types):
        for j in range(n_bundles):
            A_eq[i, i * n_bundles + j] = 1

    # 6. Inequality Constraints
    # A_ub * vars <= b_ub
    
    # IR: p[i] <= val[i] . x[i]  =>  p[i] - sum_j val[i,j]*x[i,j] <= 0
    # IC: u[i] >= u[j]  =>  val[i].x[i] - p[i] >= val[i].x[j] - p[j]
    #     => p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0
    
    n_ineq = n_types + n_types * (n_types - 1)
    A_ub = np.zeros((n_ineq, 80))
    b_ub = np.zeros(n_ineq)
    
    row = 0
    # IR Constraints
    for i in range(n_types):
        A_ub[row, 64 + i] = 1  # p[i]
        for j in range(n_bundles):
            A_ub[row, i * n_bundles + j] = -val[i, j]  # -x[i,j] * val[i,j]
        b_ub[row] = 0
        row += 1
        
    # IC Constraints
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            # p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0
            A_ub[row, 64 + i] = 1   # p[i]
            A_ub[row, 64 + j] = -1  # -p[j]
            for k in range(n_bundles):
                # + val[i,k] * x[j,k]
                A_ub[row, j * n_bundles + k] += val[i, k]
                # - val[i,k] * x[i,k]
                A_ub[row, i * n_bundles + k] -= val[i, k]
            b_ub[row] = 0
            row += 1

    bounds = [(0, None)] * 80

    # 7. Solve LP
    x_opt = None
    p_opt = None
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        
        if result.status == 0:
            x_opt = result.x[:64].reshape(n_types, n_bundles)
            p_opt = result.x[64:80]
            
            # Clean up numerical noise
            x_opt = np.maximum(x_opt, 0)
            p_opt = np.maximum(p_opt, 0)
            
            # Normalize allocations just in case
            for i in range(n_types):
                s = x_opt[i].sum()
                if s > 1e-9:
                    x_opt[i] /= s
                else:
                    x_opt[i] = np.zeros(n_bundles)
                    x_opt[i, 0] = 1.0 # Bundle (0,0)
            
            # Verify and repair if needed
            if time.time() - start_time < DEADLINE - 5:
                x_opt, p_opt = _verify_and_repair(x_opt, p_opt, val, probs)
                
        else:
            pass
    except Exception:
        pass

    # 8. If LP succeeded, convert and return
    if x_opt is not None and p_opt is not None:
        allocations = np.zeros((n_types, 2))
        for i in range(n_types):
            for j, (x1, x2) in enumerate(bundles):
                allocations[i, 0] += x1 * x_opt[i, j]
                allocations[i, 1] += x2 * x_opt[i, j]
        
        return {
            'allocations': allocations,
            'payments': p_opt
        }

    # 9. Fallback: Simple Threshold Mechanism with proper payments
    # For uniform distribution over {0,1,2,3}, use reserve r=2
    # Allocate good j if v_j >= 2
    # Payment: critical value (for uniform discrete, payment = min(v_j, r) if v_j >= r)
    # Actually for a single item with reserve r, payment = v if v >= r, else 0 is NOT IC
    # The correct payment for a second-price-like mechanism with reserve:
    # If allocated, payment = max(reserve, second highest bid) but with 1 bidder, 
    # the critical value is the reserve.
    # For a single bidder with uniform [0,3], the Myerson payment when v >= r is:
    # p(v) = r * (v/r) ... actually let's just use p(v) = v for v >= r (this is IR but not optimal)
    # Safer: use p(v) = r for v >= r (this is IR and IC for monotone allocation)
    
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    reserve = 2
    for i, (v1, v2) in enumerate(types):
        alloc1 = 1 if v1 >= reserve else 0
        alloc2 = 1 if v2 >= reserve else 0
        allocations[i, 0] = alloc1
        allocations[i, 1] = alloc2
        # Payment: sum of critical values for each allocated good
        # For good j, if v_j >= reserve, payment contribution = reserve (critical value)
        # This is IR: v_j - reserve >= 0 when v_j >= reserve
        # This is IC: monotone allocation with payment = critical value
        pay1 = reserve if v1 >= reserve else 0
        pay2 = reserve if v2 >= reserve else 0
        payments[i] = pay1 + pay2
    
    return {
        'allocations': allocations,
        'payments': payments
    }

def _verify_and_repair(x_opt, p_opt, val, probs):
    """Verify IC/IR constraints and repair numerical violations."""
    n_types = x_opt.shape[0]
    n_bundles = x_opt.shape[1]
    
    # Check and fix IR: p[i] <= val[i] @ x_opt[i]
    for i in range(n_types):
        max_pay = val[i] @ x_opt[i]
        if p_opt[i] > max_pay + 1e-6:
            p_opt[i] = max(max_pay, 0)
    
    # Check and fix IC: for all i, j: val[i]@x_opt[i] - p_opt[i] >= val[i]@x_opt[j] - p_opt[j]
    # => p_opt[i] - p_opt[j] <= val[i]@x_opt[i] - val[i]@x_opt[j]
    # We need to adjust payments to satisfy this
    # Simple approach: iteratively clamp payments
    for iteration in range(10):
        changed = False
        for i in range(n_types):
            for j in range(n_types):
                if i == j:
                    continue
                lhs = p_opt[i] - p_opt[j]
                rhs = val[i] @ x_opt[i] - val[i] @ x_opt[j]
                if lhs > rhs + 1e-6:
                    # Need to reduce p_opt[i] or increase p_opt[j]
                    # Reduce p_opt[i] by the violation amount
                    violation = lhs - rhs
                    p_opt[i] = max(p_opt[i] - violation, 0)
                    changed = True
        if not changed:
            break
    
    # Re-check IR after IC repair
    for i in range(n_types):
        max_pay = val[i] @ x_opt[i]
        if p_opt[i] > max_pay + 1e-6:
            p_opt[i] = max(max_pay, 0)
    
    return x_opt, p_opt
# EVOLVE-BLOCK-END
