"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: Specification of the value distribution
        support: Support of the value space
        constraints: Additional constraints
        
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10 second safety margin
    
    # Parse the support and distribution
    values_per_good = [0, 1, 2, 3]
    all_reports = list(product(values_per_good, values_per_good))
    
    # Map report to index
    report_to_idx = {r: i for i, r in enumerate(all_reports)}
    
    # Parse distribution_spec to get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(r, 0.0) for r in all_reports])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            pass
        else:
            probs = np.array([0.0 for _ in range(16)])
            if isinstance(distribution_spec, dict):
                 probs = np.array([distribution_spec.get(r, 0.0) for r in all_reports])
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(16) / 16.0
        
    # Fallback if time is almost up
    if time.time() > deadline:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    # Strategy:
    # 1. The payment identity for a truthful mechanism is p(v) = v.x(v) - C(v), 
    #    where C(v) is the constant of integration determined by the "critical type" 
    #    (the type where x(v)=0 or the boundary).
    # 2. For a single bidder, IC is equivalent to the allocation rule being monotone.
    # 3. We can parameterize the mechanism by the allocation probabilities x_i(t) for each type t.
    # 4. However, a more robust structural approach for finite support is to solve an LP 
    #    directly on the allocation probabilities x_{t,1}, x_{t,2} subject to:
    #       - 0 <= x <= 1
    #       - Monotonicity constraints (IC)
    #       - IR constraints (p(t) <= v(t).x(t))
    #       - Non-negative payments (p(t) >= 0)
    #    Objective: Maximize sum_t Pr(t) * p(t) = sum_t Pr(t) * (v(t).x(t) - C(t))
    #    Wait, p(t) is determined by x. 
    #    Actually, the standard formulation is:
    #    Max sum_t Pr(t) [ v(t).x(t) - E_{t' < t} [ v(t').x(t') ] ] ? No, that's for 1D.
    
    # Let's use the direct LP formulation for payments and allocations.
    # Variables: x_{t,1}, x_{t,2}, p_t for all t in T (16 types). Total 48 vars.
    # Constraints:
    # 1. IC: For all t, t', 
    #    p(t) - v(t).x(t) >= p(t') - v(t).x(t')  (No over-reporting)
    #    p(t) - v(t').x(t) >= p(t') - v(t').x(t') (No under-reporting)
    #    Actually, the standard IC constraint is:
    #    v(t).x(t) - p(t) >= v(t).x(t') - p(t') for all t, t'.
    #    This is 16*16 = 256 constraints.
    # 2. IR: v(t).x(t) - p(t) >= 0 for all t. (16 constraints)
    # 3. Non-negative payments: p(t) >= 0 for all t. (16 constraints)
    # 4. Feasible allocations: 0 <= x_{t,i} <= 1. (32 constraints)
    
    # Objective: Maximize sum_t Pr(t) * p(t)
    # This is a Linear Program.
    # Let's set it up using scipy.optimize.linprog.
    
    n_types = 16
    n_goods = 2
    n_vars = n_types * n_goods + n_types # x_1...x_16, p_1...p_16
    
    # Indices:
    # x_{t,i} index: t * 2 + i
    # p_t index: n_types * 2 + t
    
    def get_x_idx(t, i):
        return t * 2 + i
    
    def get_p_idx(t):
        return n_types * 2 + t

    # Objective: Max sum_t Pr(t) * p_t
    # linprog minimizes, so we minimize -sum_t Pr(t) * p_t
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[get_p_idx(t)] = -probs[t]
        
    # Constraints A_ub x <= b_ub
    
    A_rows = []
    b_rows = []
    
    # 1. IC Constraints: v(t).x(t) - p(t) >= v(t).x(t') - p(t')
    # Rearranged: -v(t).x(t) + p(t) + v(t).x(t') - p(t') <= 0
    # Variables involved: x_t, p_t, x_t', p_t'
    
    for t in range(n_types):
        v_t = np.array(all_reports[t], dtype=float)
        for tp in range(n_types):
            if t == tp:
                continue
            v_tp = np.array(all_reports[tp], dtype=float)
            
            row = np.zeros(n_vars)
            
            # -v(t).x(t)
            for i in range(n_goods):
                row[get_x_idx(t, i)] -= v_t[i]
            
            # +p(t)
            row[get_p_idx(t)] += 1.0
            
            # +v(t).x(t')
            for i in range(n_goods):
                row[get_x_idx(tp, i)] += v_t[i]
                
            # -p(t')
            row[get_p_idx(tp)] -= 1.0
            
            A_rows.append(row)
            b_rows.append(0.0)
            
    # 2. IR Constraints: v(t).x(t) - p(t) >= 0
    # Rearranged: -v(t).x(t) + p(t) <= 0
    for t in range(n_types):
        v_t = np.array(all_reports[t], dtype=float)
        row = np.zeros(n_vars)
        for i in range(n_goods):
            row[get_x_idx(t, i)] -= v_t[i]
        row[get_p_idx(t)] += 1.0
        A_rows.append(row)
        b_rows.append(0.0)
        
    # 3. Non-negative payments: p(t) >= 0
    # This is handled by bounds.
    
    # 4. Feasible allocations: 0 <= x <= 1
    # This is handled by bounds.
    
    # Bounds for linprog
    bounds = []
    for t in range(n_types):
        for i in range(n_goods):
            bounds.append((0, 1)) # x_{t,i}
    for t in range(n_types):
        bounds.append((0, None)) # p_t
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            # Extract solution
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
            for t in range(n_types):
                for i in range(n_goods):
                    allocations[t, i] = res.x[get_x_idx(t, i)]
                payments[t] = res.x[get_p_idx(t)]
                
            # Verify constraints with tolerance
            # IC
            max_regret = 0.0
            for t in range(n_types):
                v_t = np.array(all_reports[t], dtype=float)
                util_t = v_t @ allocations[t] - payments[t]
                for tp in range(n_types):
                    util_tp_as_t = v_t @ allocations[tp] - payments[tp]
                    regret = util_tp_as_t - util_t
                    if regret > max_regret:
                        max_regret = regret
            
            # IR
            min_ir = np.inf
            for t in range(n_types):
                v_t = np.array(all_reports[t], dtype=float)
                util_t = v_t @ allocations[t] - payments[t]
                if util_t < min_ir:
                    min_ir = util_t
            
            # Non-negative payments
            min_pay = np.min(payments)
            
            if max_regret <= 1e-8 and min_ir >= -1e-8 and min_pay >= -1e-8:
                return {'allocations': allocations, 'payments': payments}
            else:
                # If LP failed validation due to numerical issues, try to fix or fallback
                # Simple fallback: Zero mechanism
                pass
        else:
            # LP failed
            pass
            
    except Exception as e:
        pass

    # Fallback: Simple threshold mechanism (same as seed but slightly optimized)
    best_revenue = -np.inf
    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    
    thresholds = [0, 1, 2, 3]
    
    for t1 in thresholds:
        for t2 in thresholds:
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
            for i, (v1, v2) in enumerate(all_reports):
                alloc1 = 1.0 if v1 >= t1 else 0.0
                alloc2 = 1.0 if v2 >= t2 else 0.0
                allocations[i, 0] = alloc1
                allocations[i, 1] = alloc2
                
                payment = 0.0
                if alloc1 > 0:
                    payment += t1
                if alloc2 > 0:
                    payment += t2
                payments[i] = payment
            
            ir_violated = False
            for i, (v1, v2) in enumerate(all_reports):
                value = v1 * allocations[i, 0] + v2 * allocations[i, 1]
                if payments[i] > value + 1e-8:
                    ir_violated = True
                    break
            
            if ir_violated:
                continue
            
            if np.any(payments < -1e-8):
                continue
                
            # Check IC for this threshold mechanism
            # Threshold mechanisms are generally IC if monotone.
            # Let's verify regret to be safe.
            valid_ic = True
            for t in range(16):
                v_t = np.array(all_reports[t], dtype=float)
                util_t = v_t @ allocations[t] - payments[t]
                for tp in range(16):
                    util_tp_as_t = v_t @ allocations[tp] - payments[tp]
                    if util_tp_as_t > util_t + 1e-8:
                        valid_ic = False
                        break
                if not valid_ic:
                    break
            
            if not valid_ic:
                continue

            expected_revenue = np.sum(probs * payments)
            
            if expected_revenue > best_revenue:
                best_revenue = expected_revenue
                best_allocations = allocations.copy()
                best_payments = payments.copy()
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_revenue <= 0:
        best_allocations = np.ones((16, 2))
        best_payments = np.zeros(16)
        
    return {'allocations': best_allocations, 'payments': best_payments}
# EVOLVE-BLOCK-END
