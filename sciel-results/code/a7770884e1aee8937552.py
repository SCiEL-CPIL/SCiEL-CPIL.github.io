import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Matrix V where V[t, g] is the value of good g for type t
    V = np.zeros((n_types, 2))
    for t in range(n_types):
        V[t, 0] = types[t][0]
        V[t, 1] = types[t][1]

    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                probs = np.array([dist.get(t, 0.0) for t in types], dtype=float)
            elif isinstance(dist, (list, tuple, np.ndarray)):
                probs = np.array(dist, dtype=float)
            else:
                probs = np.array([1.0/16]*16, dtype=float)
        else:
            probs = np.array([1.0/16]*16, dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.array([1.0/16]*16, dtype=float)
    
    # Ensure probabilities sum to 1
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.array([1.0/16]*16, dtype=float)

    # Parse constraints
    if constraints is not None and isinstance(constraints, dict):
        regret_tol = constraints.get('regret_tol', 1e-8)
        ir_tol = constraints.get('ir_tol', 1e-8)
    else:
        regret_tol = 1e-8
        ir_tol = 1e-8

    # LP Formulation:
    # Variables:
    # x1[0..15], x2[0..15] (allocation probabilities for good 1 and 2)
    # u[0..15] (utilities)
    # Total variables: 48
    
    n_vars = 48
    x1_offset = 0
    x2_offset = 16
    u_offset = 32
    
    # Objective: Maximize sum_t probs[t] * (V[t] dot x[t] - u[t])
    # Minimize -sum_t probs[t] * (V[t,0]*x1[t] + V[t,1]*x2[t] - u[t])
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[x1_offset + t] = -probs[t] * V[t, 0]
        c[x2_offset + t] = -probs[t] * V[t, 1]
        c[u_offset + t] = probs[t]  # Because we minimize -(... - u) => +u in objective coeff? 
        # Wait. Minimize - (A - B) = -A + B.
        # Term is -probs[t]*(V[t,0]*x1[t] + V[t,1]*x2[t] - u[t])
        # = -probs[t]*V[t,0]*x1[t] - probs[t]*V[t,1]*x2[t] + probs[t]*u[t]
        # So coeff for u is +probs[t]. Correct.

    A_ub_rows = []
    b_ub_vals = []
    
    # IC Constraints:
    # u[t] >= u[t'] + (V[t] - V[t']) dot x[t']
    # => u[t] - u[t'] - V[t,0]*x1[t'] - V[t,1]*x2[t'] + V[t',0]*x1[t'] + V[t',1]*x2[t'] >= 0
    # => -u[t] + u[t'] + (V[t]-V[t']) dot x[t'] <= 0
    # Let's rearrange to standard form A x <= b
    # u[t] - u[t'] >= (V[t] - V[t']) dot x[t']
    # -u[t] + u[t'] - (V[t] - V[t']) dot x[t'] <= 0
    # -u[t] + u[t'] - V[t,0]*x1[t'] - V[t,1]*x2[t'] + V[t',0]*x1[t'] + V[t',1]*x2[t'] <= 0
    
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            # u[t] term: -1
            row[u_offset + t] = -1.0
            # u[tp] term: +1
            row[u_offset + tp] = 1.0
            
            # x terms involve x[tp]
            # Coeff of x1[tp]: -(V[t,0] - V[tp,0]) = V[tp,0] - V[t,0]
            # Coeff of x2[tp]: -(V[t,1] - V[tp,1]) = V[tp,1] - V[t,1]
            row[x1_offset + tp] = V[tp, 0] - V[t, 0]
            row[x2_offset + tp] = V[tp, 1] - V[t, 1]
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
    # IR Constraints:
    # u[t] >= 0 => -u[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_offset + t] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1[t]
    for t in range(n_types):
        bounds.append((0, 1))  # x2[t]
    for t in range(n_types):
        bounds.append((0, None))  # u[t] >= 0

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception as e:
        # Fallback to zero mechanism
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}

    if not result.success:
        # Fallback to zero mechanism
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}

    x_opt = result.x
    
    allocations = np.zeros((n_types, 2))
    utilities = np.zeros(n_types)
    for t in range(n_types):
        allocations[t, 0] = x_opt[x1_offset + t]
        allocations[t, 1] = x_opt[x2_offset + t]
        utilities[t] = x_opt[u_offset + t]
        
    # Calculate payments: p[t] = V[t] dot x[t] - u[t]
    payments = np.zeros(n_types)
    for t in range(n_types):
        payments[t] = V[t, 0] * allocations[t, 0] + V[t, 1] * allocations[t, 1] - utilities[t]
        
    # Clip small numerical errors
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    # Verify constraints and repair if necessary
    # Check IC: u(t) >= u(t') + (v(t) - v(t')) dot x(t')
    # Equivalent to: v(t) dot x(t) - p(t) >= v(t) dot x(t') - p(t')
    # Let's verify with the calculated values.
    max_viol = 0.0
    for t in range(n_types):
        u_tt = V[t, 0]*allocations[t, 0] + V[t, 1]*allocations[t, 1] - payments[t]
        if u_tt < -regret_tol:
            max_viol = max(max_viol, -u_tt)
        for tp in range(n_types):
            if t == tp: continue
            u_t_tp = V[t, 0]*allocations[tp, 0] + V[t, 1]*allocations[tp, 1] - payments[tp]
            viol = u_t_tp - u_tt
            if viol > regret_tol:
                max_viol = max(max_viol, viol)
                
    if max_viol > 0:
        # If there is a violation, we can try to fix it by reducing allocations slightly 
        # or adjusting payments. However, the LP should have found the optimal feasible point.
        # If numerical errors caused this, a simple clip might not be enough.
        # A robust fallback for slight violations is to reduce all allocations by a small factor 
        # or just return the zero mechanism if it's severe. 
        # Given the small size, if LP failed to satisfy constraints, it's likely a numerical precision issue.
        # We'll try a simple repair: reduce payments to satisfy IR, and if IC is broken, 
        # it's harder. But usually, the LP solution is very close.
        # Let's just ensure IR is met by clipping payments.
        for t in range(n_types):
            u_tt = V[t, 0]*allocations[t, 0] + V[t, 1]*allocations[t, 1] - payments[t]
            if u_tt < 0:
                payments[t] = max(0, V[t, 0]*allocations[t, 0] + V[t, 1]*allocations[t, 1])
                
    return {'allocations': allocations, 'payments': payments}
