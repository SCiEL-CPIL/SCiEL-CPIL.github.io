# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def _verify_and_repair(allocations, payments, val, probs, n_types, n_bundles, types, bundles):
    """
    Verify IC and IR constraints.
    Returns (allocations, payments, is_valid)
    """
    a1 = allocations[:, 0]
    a2 = allocations[:, 1]
    
    v1 = np.array([t[0] for t in types])
    v2 = np.array([t[1] for t in types])
    
    # Expected value for type i given allocation a1, a2
    # E[val[i]·x] = v1_i * a1 + v2_i * a2
    expected_val = v1 * a1 + v2 * a2
    utilities = expected_val - payments
    
    # Check IR
    ir_violation = np.max(-utilities) if n_types > 0 else 0
    
    # Check IC
    ic_violation = 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            # u(i, x_j, p_j) = v1_i*a1[j] + v2_i*a2[j] - p[j]
            u_i_j = v1[i] * a1[j] + v2[i] * a2[j] - payments[j]
            u_i_i = utilities[i]
            if u_i_j > u_i_i + ic_violation:
                ic_violation = u_i_j - u_i_i
    
    max_violation = max(ir_violation, ic_violation)
    
    if max_violation > 1e-8:
        return allocations, payments, False
    
    return allocations, payments, True

def _polish_payments(x_opt, val, probs, n_types, n_bundles):
    """
    Given a fixed allocation x_opt (probabilities over bundles for each type),
    find the optimal payments p that satisfy IC and IR and maximize revenue.
    """
    # Variables: p[i] for i in 0..15
    n_vars = n_types
    c = -probs # Maximize sum probs[i]*p[i] => Minimize -sum probs[i]*p[i]
    
    # Constraints:
    # 1. IR: p[i] <= val[i] . x_opt[i]
    #    p[i] - val[i] . x_opt[i] <= 0
    # 2. IC: u(i, x_i, p_i) >= u(i, x_j, p_j)
    #    val[i].x_i - p_i >= val[i].x_j - p_j
    #    p_i - p_j <= val[i].x_i - val[i].x_j
    # 3. p[i] >= 0 (handled by bounds)
    
    n_ineq = n_types + n_types * (n_types - 1)
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    row = 0
    # IR Constraints
    for i in range(n_types):
        A_ub[row, i] = 1
        b_ub[row] = val[i] @ x_opt[i]
        row += 1
        
    # IC Constraints
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            # p_i - p_j <= val[i].x_i - val[i].x_j
            A_ub[row, i] = 1
            A_ub[row, j] = -1
            b_ub[row] = val[i] @ x_opt[i] - val[i] @ x_opt[j]
            row += 1
            
    bounds = [(0, None)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x
    except Exception:
        pass
        
    # Fallback: 0 payments
    return np.zeros(n_types)

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
    
    n_vars = 80
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # Minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[64 + i] = -probs[i]

    # 5. Equality Constraints
    # Sum_j x[i, j] = 1 for all i
    A_eq = np.zeros((n_types, n_vars))
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
    A_ub = np.zeros((n_ineq, n_vars))
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

    bounds = [(0, None)] * n_vars

    # 7. Solve LP
    x_opt = None
    p_opt = None
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        
        if result.status == 0:
            x_opt = result.x[:64].reshape(n_types, n_bundles)
            p_opt = result.x[64:80]
            
            # Clean up numerical noise in x
            x_opt = np.maximum(x_opt, 0)
            
            # Normalize allocations
            for i in range(n_types):
                s = x_opt[i].sum()
                if s > 1e-9:
                    x_opt[i] /= s
                else:
                    x_opt[i] = np.zeros(n_bundles)
                    x_opt[i, 0] = 1.0 # Bundle (0,0)
                    
            # Snap x to clean values to help polish step
            # If x is very close to 0 or 1, snap it
            for i in range(n_types):
                for j in range(n_bundles):
                    if x_opt[i, j] < 1e-6:
                        x_opt[i, j] = 0
                    elif x_opt[i, j] > 1 - 1e-6:
                        x_opt[i, j] = 1
                # Renormalize after snapping
                s = x_opt[i].sum()
                if s > 0:
                    x_opt[i] /= s
                else:
                    x_opt[i, 0] = 1.0

    except Exception:
        pass

    if x_opt is not None:
        # 8. Polish Payments
        # Use the fixed x_opt to find optimal p
        p_opt = _polish_payments(x_opt, val, probs, n_types, n_bundles)
        
        # Convert bundle probabilities to expected allocation per good
        allocations = np.zeros((n_types, 2))
        for i in range(n_types):
            for j, (x1, x2) in enumerate(bundles):
                allocations[i, 0] += x1 * x_opt[i, j]
                allocations[i, 1] += x2 * x_opt[i, j]
        
        # Verify constraints
        allocations, p_opt, is_valid = _verify_and_repair(
            allocations, p_opt, val, probs, n_types, n_bundles, types, bundles
        )
        
        if is_valid:
            return {
                'allocations': allocations,
                'payments': p_opt
            }
        else:
            # If polish failed or verification failed, try original p_opt from LP
            p_opt_orig = result.x[64:80]
            p_opt_orig = np.maximum(p_opt_orig, 0)
            allocations, p_opt_orig, is_valid_orig = _verify_and_repair(
                allocations, p_opt_orig, val, probs, n_types, n_bundles, types, bundles
            )
            if is_valid_orig:
                return {
                    'allocations': allocations,
                    'payments': p_opt_orig
                }
            # Else fall through to fallback

    # 9. Fallback: Zero mechanism
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
