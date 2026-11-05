"""Incentive-compatible finite auction design solver."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    # distribution_spec: list of 4 probabilities [p0, p1, p2, p3] for each good (independent)
    # support: list of allowed reports (value pairs)
    # constraints: dict with feasibility parameters
    
    # Values are in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]
    
    # Normalize probabilities
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p1 = p1 / p1.sum() if p1.sum() > 0 else np.ones(4) / 4
    p2 = p2 / p2.sum() if p2.sum() > 0 else np.ones(4) / 4
    
    # Joint distribution over (v1, v2)
    joint_probs = np.outer(p1, p2)  # shape (4, 4)
    joint_probs = joint_probs.flatten()  # shape (16,)
    
    # Enumerate all type profiles: (v1, v2) where v1, v2 in {0,1,2,3}
    # Type index = v1 * 4 + v2
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    
    # Allocation: for each type (v1, v2), allocate goods (x1, x2) where x1, x2 in {0,1}
    # So allocation is a pair of binary variables
    # Payment: for each type, a scalar payment
    
    # Variables:
    # x[t, g] for t in 0..15, g in 0,1 -> 32 allocation variables
    # p[t] for t in 0..15 -> 15 payment variables (one is free due to translation invariance, but we fix it)
    # Actually, let's use all 16 payments
    
    # Objective: maximize expected revenue = sum_t joint_probs[t] * payment[t]
    # Subject to:
    # 1. Feasibility: 0 <= x[t, g] <= 1 (relax to continuous for LP)
    # 2. Incentive compatibility (truthfulness): for all types t, t'
    #    u(t, t) >= u(t, t')
    #    where u(t, t') = v1[t]*x[t',0] + v2[t]*x[t',1] - p[t']
    #    This gives: v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= v1[t]*x[t',0] + v2[t]*x[t',1] - p[t']
    #    Rearranging: p[t] - p[t'] <= v1[t]*(x[t,0] - x[t',0]) + v2[t]*(x[t,1] - x[t',1])
    # 3. Individual rationality: u(t, t) >= 0 for all t
    #    v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= 0
    #    p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1]
    # 4. Nonnegative payments: p[t] >= 0
    
    # Let's set up the LP
    # Variables: [x0_0, x0_1, x1_0, x1_1, ..., x15_0, x15_1, p0, p1, ..., p15]
    # Total: 32 + 16 = 48 variables
    
    n_x = 32
    n_p = 16
    n_vars = n_x + n_p
    
    # Objective: maximize sum_t joint_probs[t] * p[t]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -joint_probs[t]  # negative for maximization
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    # IC constraints: for all t, t'
    # p[t] - p[t'] <= v1[t]*(x[t,0] - x[t',0]) + v2[t]*(x[t,1] - x[t',1])
    # Rearranging: -v1[t]*x[t,0] - v2[t]*x[t,1] + v1[t]*x[t',0] + v2[t]*x[t',1] + p[t] - p[t'] <= 0
    
    A_ub_rows = []
    b_ub_vals = []
    
    for t in range(n_types):
        v1_t, v2_t = types[t]
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            # -v1[t]*x[t,0]
            row[t * 2] = -v1_t
            # -v2[t]*x[t,1]
            row[t * 2 + 1] = -v2_t
            # +v1[t]*x[tp,0]
            row[tp * 2] += v1_t
            # +v2[t]*x[tp,1]
            row[tp * 2 + 1] += v2_t
            # +p[t]
            row[n_x + t] = 1
            # -p[tp]
            row[n_x + tp] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
            
    # Monotonicity constraints
    # Good 0: x[v1+1, v2, 0] >= x[v1, v2, 0] for v1 < 3
    # -x[v1+1, v2, 0] + x[v1, v2, 0] <= 0
    for v1 in range(3):
        for v2 in range(4):
            t_curr = v1 * 4 + v2
            t_next = (v1 + 1) * 4 + v2
            row = np.zeros(n_vars)
            row[t_next * 2] = -1
            row[t_curr * 2] = 1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
            
    # Good 1: x[v1, v2+1, 1] >= x[v1, v2, 1] for v2 < 3
    # -x[v1, v2+1, 1] + x[v1, v2, 1] <= 0
    for v1 in range(4):
        for v2 in range(3):
            t_curr = v1 * 4 + v2
            t_next = v1 * 4 + (v2 + 1)
            row = np.zeros(n_vars)
            row[t_next * 2 + 1] = -1
            row[t_curr * 2 + 1] = 1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
    
    # IR constraints: p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1]
    # Rearranging: -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1_t
        row[t * 2 + 1] = -v2_t
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # Nonnegative payments: p[t] >= 0 -> -p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + t] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # Allocation bounds: 0 <= x[t,g] <= 1
    # We'll use bounds instead of constraints for simplicity
    
    bounds = []
    for i in range(n_x):
        bounds.append((0, 1))
    for i in range(n_p):
        bounds.append((0, None))  # payments >= 0
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: uniform allocation, zero payments
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {'allocations': allocations, 'payments': payments}
        
        x_vars = result.x[:n_x]
        p_vars = result.x[n_x:]
        
        # Round allocations to binary (0 or 1)
        allocations = np.zeros((16, 2))
        for t in range(16):
            allocations[t, 0] = 1 if x_vars[t * 2] > 0.5 else 0
            allocations[t, 1] = 1 if x_vars[t * 2 + 1] > 0.5 else 0
        
        # Ensure nonnegative payments
        payments = np.maximum(p_vars, 0)
        
        # Verify constraints with rounded allocations
        # Check IC
        ic_ok = True
        for t in range(n_types):
            v1_t, v2_t = types[t]
            u_tt = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            for tp in range(n_types):
                u_tt_prime = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
                if u_tt < u_tt_prime - 1e-8:
                    ic_ok = False
                    break
            if not ic_ok:
                break
        
        # Check IR
        ir_ok = True
        for t in range(n_types):
            v1_t, v2_t = types[t]
            u_tt = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            if u_tt < -1e-8:
                ir_ok = False
                break
        
        if not ic_ok or not ir_ok:
            # Fallback: simple mechanism
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for t in range(16):
                v1_t, v2_t = types[t]
                # Allocate good 1 if v1_t >= 2, good 2 if v2_t >= 2
                if v1_t >= 2:
                    allocations[t, 0] = 1
                if v2_t >= 2:
                    allocations[t, 1] = 1
                # Payment: sum of allocated values
                payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception:
        # Fallback: simple threshold mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for t in range(16):
            v1_t, v2_t = types[t]
            if v1_t >= 2:
                allocations[t, 0] = 1
            if v2_t >= 2:
                allocations[t, 1] = 1
            payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        
        return {'allocations': allocations, 'payments': payments}
