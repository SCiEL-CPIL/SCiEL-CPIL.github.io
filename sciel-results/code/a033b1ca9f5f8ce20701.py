"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 60.0  # seconds, with safety margin
    
    # Parse the problem
    # support: list of (v1, v2) tuples for all types
    # distribution_spec: probabilities for each type in support
    types = list(support)
    n_types = len(types)
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(t, 0.0) for t in types])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        # Try to parse
        probs = np.array([distribution_spec[t] for t in types], dtype=float)
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    
    # Values for each type
    values = np.array([t for t in types], dtype=float)  # shape (n_types, 2)
    
    # Decision variables:
    # x_i_j: probability of allocating good j to type i (j=0,1)
    # p_i: payment from type i
    # Total variables: 2*n_types + n_types = 3*n_types
    
    # Revenue = sum_i probs[i] * p_i
    # Maximize revenue
    
    # Constraints:
    # 1. 0 <= x_i_j <= 1 for all i, j
    # 2. p_i >= 0
    # 3. IR: p_i <= sum_j values[i][j] * x_i_j
    #    => p_i - sum_j values[i][j] * x_i_j <= 0
    # 4. IC: For all i, i': 
    #    v_i . x_i - p_i >= v_i . x_i' - p_i'
    #    => p_i - v_i . x_i <= p_i' - v_i . x_i'
    #    => p_i - p_i' - v_i . x_i + v_i . x_i' <= 0
    # 5. Regret constraint: for all i, i':
    #    v_i . x_i - p_i >= v_i . x_i' - p_i'  (same as IC)
    #    The regret is max over i' of [v_i . x_i' - p_i' - (v_i . x_i - p_i)]
    #    We need regret <= 1e-8, which is the same as IC
    
    # Let's set up the LP:
    # Variables: [x_0_0, x_0_1, p_0, x_1_0, x_1_1, p_1, ..., x_{n-1}_0, x_{n-1}_1, p_{n-1}]
    # Index for type i, good j: 3*i + j
    # Index for type i, payment: 3*i + 2
    
    n_vars = 3 * n_types
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[3*i + 2] = -probs[i]  # payment variable
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    A_rows = []
    b_rows = []
    
    # IR constraints: p_i - sum_j v_ij * x_ij <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[3*i + 2] = 1.0  # p_i
        for j in range(2):
            row[3*i + j] = -values[i][j]  # -v_ij * x_ij
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC constraints: for all i, i' (i != i'):
    # p_i - p_i' - v_i . x_i + v_i . x_i' <= 0
    for i in range(n_types):
        for ip in range(n_types):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            # p_i
            row[3*i + 2] = 1.0
            # -p_i'
            row[3*ip + 2] = -1.0
            # -v_i . x_i
            for j in range(2):
                row[3*i + j] = -values[i][j]
            # +v_i . x_i'
            for j in range(2):
                row[3*ip + j] = values[i][j]
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x_ij <= 1, p_i >= 0
    bounds = []
    for i in range(n_types):
        for j in range(2):
            bounds.append((0, 1))  # x_ij
        bounds.append((0, None))  # p_i
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: uniform allocation with zero payment
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {"allocations": allocations, "payments": payments}
        
        x = result.x
        
        # Extract allocations and payments
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for i in range(n_types):
            allocations[i][0] = x[3*i + 0]
            allocations[i][1] = x[3*i + 1]
            payments[i] = x[3*i + 2]
        
        # Clip to ensure feasibility
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        
        # Verify IR
        for i in range(n_types):
            max_payment = values[i] @ allocations[i]
            if payments[i] > max_payment + 1e-9:
                payments[i] = max_payment
        
        # Verify IC / regret
        for i in range(n_types):
            for ip in range(n_types):
                if i == ip:
                    continue
                util_i = values[i] @ allocations[i] - payments[i]
                util_i_as_ip = values[i] @ allocations[ip] - payments[ip]
                regret = util_i_as_ip - util_i
                if regret > 1e-8:
                    # This shouldn't happen if LP is solved correctly
                    # Adjust payment to fix
                    payments[i] = max(payments[i], values[i] @ allocations[ip] - values[i] @ allocations[i] + 1e-9)
        
        # Final safety check
        for i in range(n_types):
            max_payment = values[i] @ allocations[i]
            if payments[i] > max_payment:
                payments[i] = max_payment
            payments[i] = max(payments[i], 0)
        
        return {"allocations": allocations, "payments": payments}
    
    except Exception:
        # Fallback: simple threshold mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for i in range(n_types):
            v1, v2 = values[i]
            # Allocate if value is high enough
            if v1 >= 2:
                allocations[i][0] = 1.0
            if v2 >= 2:
                allocations[i][1] = 1.0
            payments[i] = 0.0
        return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
