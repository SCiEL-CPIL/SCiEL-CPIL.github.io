"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    
    # Parse distribution probabilities
    # distribution_spec should give probabilities for each type
    if isinstance(distribution_spec, dict):
        # Try to extract probabilities from dict
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to find any array-like value
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, tuple, np.ndarray)):
                    probs = np.array(distribution_spec[key], dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    if len(probs) != n_types:
        # Try to reshape or handle differently
        if len(probs) == 4:
            # Maybe it's per-good marginal
            probs = np.array([probs[i]*probs[j] for i in range(4) for j in range(4)])
        else:
            probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum()
    
    # Parse support: which reports are allowed
    # support could be a list of type indices, or a set of (v1,v2) pairs, or None (all allowed)
    if support is None:
        allowed_types = list(range(n_types))
    elif isinstance(support, (list, tuple, set, np.ndarray)):
        support = list(support)
        if len(support) > 0:
            first = support[0]
            if isinstance(first, (list, tuple)):
                # Support is a list of (v1, v2) pairs
                allowed_types = []
                for s in support:
                    idx = types.index(tuple(s))
                    allowed_types.append(idx)
            else:
                # Support is a list of indices
                allowed_types = list(support)
        else:
            allowed_types = list(range(n_types))
    else:
        allowed_types = list(range(n_types))
    
    n_allowed = len(allowed_types)
    
    # Parse constraints
    # constraints could be a dict with various constraint specifications
    # For now, we'll assume standard constraints:
    # - Feasible probabilities (allocations in [0,1])
    # - Nonnegative payments
    # - Individual rationality
    # - Regret <= 1e-8
    
    # The task says: maximize expected revenue normalized by an independently solved finite-support mechanism LP
    # So we need to solve the LP for the optimal revenue, then return a mechanism that achieves it
    
    # Formulate the mechanism design LP:
    # Variables: x_{i,j} for each type i and good j (allocation probability)
    #            p_i for each type i (payment)
    # 
    # For a single bidder with quasi-linear utility:
    # u_i = sum_j v_{i,j} * x_{i,j} - p_i
    # 
    # Incentive compatibility: for all types i, i':
    #   u_i >= u_{i'} + (v_i - v_{i'}) · x_i  (when reporting i' but type is i)
    #   More precisely: u_i >= u_{i'} + (v_i - v_{i'}) · x_{i'}
    # 
    # Actually, the IC constraint is:
    #   u_i >= u_{i'} + (v_i - v_{i'}) · x_{i'}  for all i, i'
    # 
    # Individual rationality: u_i >= 0 for all i
    # 
    # Objective: maximize sum_i probs[i] * p_i
    # 
    # But p_i = sum_j v_{i,j} * x_{i,j} - u_i
    # 
    # So objective: maximize sum_i probs[i] * (sum_j v_{i,j} * x_{i,j} - u_i)
    #              = sum_i probs[i] * sum_j v_{i,j} * x_{i,j} - sum_i probs[i] * u_i
    # 
    # Constraints:
    # 1. 0 <= x_{i,j} <= 1 for all i, j
    # 2. u_i >= 0 for all i
    # 3. IC: u_i - u_{i'} >= (v_i - v_{i'}) · x_{i'} for all i, i'
    # 
    # This is a linear program!
    
    # Let's set up the LP properly
    # Variables: x_{i,j} for i in 0..15, j in 0,1 (32 vars)
    #            u_i for i in 0..15 (16 vars)
    # Total: 48 variables
    # 
    # But we can eliminate u_i using the fact that at the optimum, 
    # the IR constraint binds for the lowest type (type 0,0) if values are monotone
    # 
    # Actually, let's just solve the full LP with scipy.optimize.linprog
    
    # Variable ordering:
    # x_{0,0}, x_{0,1}, x_{1,0}, x_{1,1}, ..., x_{15,0}, x_{15,1}, u_0, u_1, ..., u_{15}
    # Indices:
    # x_{i,j} -> 2*i + j
    # u_i -> 32 + i
    
    n_x = 2 * n_types  # 32
    n_u = n_types  # 16
    n_vars = n_x + n_u  # 48
    
    # Objective: maximize sum_i probs[i] * (sum_j v_{i,j} * x_{i,j} - u_i)
    # linprog minimizes, so we negate:
    # minimize -sum_i probs[i] * sum_j v_{i,j} * x_{i,j} + sum_i probs[i] * u_i
    
    c = np.zeros(n_vars)
    for i in range(n_types):
        for j in range(2):
            c[2*i + j] = -probs[i] * types[i][j]
        c[32 + i] = probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 
    # IC constraints: for all i, i' (i != i'):
    #   u_i - u_{i'} >= (v_i - v_{i'}) · x_{i'}
    #   => -(v_i - v_{i'}) · x_{i'} - u_i + u_{i'} <= 0
    #   => -(v_{i,0} - v_{i',0}) * x_{i',0} - (v_{i,1} - v_{i',1}) * x_{i',1} - u_i + u_{i'} <= 0
    # 
    # IR constraints: u_i >= 0 => -u_i <= 0
    # 
    # Feasibility: 0 <= x_{i,j} <= 1, handled by bounds
    
    # Build A_ub and b_ub
    # Number of IC constraints: 16*15 = 240
    # Number of IR constraints: 16
    # Total: 256
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_constraints = n_ic + n_ir
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    for i in range(n_types):
        for ip in range(n_types):
            if i == ip:
                continue
            # IC: u_i - u_{ip} >= (v_i - v_{ip}) · x_{ip}
            # => -(v_i - v_{ip}) · x_{ip} - u_i + u_{ip} <= 0
            A_ub[row, 2*ip + 0] = -(types[i][0] - types[ip][0])
            A_ub[row, 2*ip + 1] = -(types[i][1] - types[ip][1])
            A_ub[row, 32 + i] = -1
            A_ub[row, 32 + ip] = 1
            b_ub[row] = 0
            row += 1
    
    # IR constraints: -u_i <= 0
    for i in range(n_types):
        A_ub[row, 32 + i] = -1
        b_ub[row] = 0
        row += 1
    
    # Bounds:
    # x_{i,j} in [0, 1]
    # u_i in [0, inf)  (but effectively bounded)
    
    bounds = []
    for i in range(n_types):
        for j in range(2):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation, zero payments (IR satisfied, IC may not be)
        # Actually, let's try a simpler approach
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_opt = result.x
    
    allocations = np.zeros((n_types, 2))
    for i in range(n_types):
        allocations[i, 0] = x_opt[2*i + 0]
        allocations[i, 1] = x_opt[2*i + 1]
    
    u_opt = x_opt[32:32+n_types]
    
    payments = np.zeros(n_types)
    for i in range(n_types):
        payments[i] = types[i][0] * allocations[i, 0] + types[i][1] * allocations[i, 1] - u_opt[i]
    
    # Ensure nonnegative payments
    payments = np.maximum(payments, 0)
    
    # Clip allocations to [0,1] for numerical safety
    allocations = np.clip(allocations, 0, 1)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
