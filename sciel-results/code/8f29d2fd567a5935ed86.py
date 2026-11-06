"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        elif 'type_probs' in distribution_spec:
            probs = np.array(distribution_spec['type_probs'], dtype=float)
        else:
            # Try to find a list/array in the dict
            for k, v in distribution_spec.items():
                if isinstance(v, (list, np.ndarray)) and len(v) == n_types:
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Parse support - the set of allowed reports/values
    # Support should be the 16 type combinations
    if support is not None and isinstance(support, (list, tuple, np.ndarray)):
        if len(support) == n_types:
            # Use as-is if it's a list of tuples
            pass
    
    # Parse constraints
    # constraints may specify regret tolerance, etc.
    regret_tol = 1e-8
    if isinstance(constraints, dict):
        if 'regret_tol' in constraints:
            regret_tol = constraints['regret_tol']
        elif 'regret' in constraints:
            regret_tol = constraints['regret']
    
    # Value matrix: V[i, j] = value of good j for type i
    V = np.array([[t[0], t[1]] for t in types], dtype=float)  # (16, 2)
    
    # LP variables:
    # x[i, j] for i in 0..15, j in 0..1: allocation probability (16*2 = 32 vars)
    # p[i] for i in 0..15: payment (16 vars)
    # Total: 48 variables
    # Variable indexing: x[i,j] = i*2 + j, p[i] = 32 + i
    
    n_x = n_types * n_goods  # 32
    n_p = n_types  # 16
    n_vars = n_x + n_p  # 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_x + i] = -probs[i]  # maximize payment
    
    # Constraints:
    # 1. IC constraints: for all i, i': 
    #    sum_j V[i,j]*x[i,j] - p[i] >= sum_j V[i,j]*x[i',j] - p[i']
    #    => sum_j V[i,j]*x[i,j] - sum_j V[i,j]*x[i',j] - p[i] + p[i'] >= 0
    #    In linprog form (A_ub @ x <= b_ub), we negate:
    #    -sum_j V[i,j]*x[i,j] + sum_j V[i,j]*x[i',j] + p[i] - p[i'] <= 0
    
    # 2. IR constraints: for all i:
    #    sum_j V[i,j]*x[i,j] - p[i] >= 0
    #    => -sum_j V[i,j]*x[i,j] + p[i] <= 0
    
    # 3. Allocation feasibility: 0 <= x[i,j] <= 1
    # 4. Payment non-negativity: p[i] >= 0
    
    # Build constraint matrix for IC and IR
    # IC: n_types * n_types = 256 constraints
    # IR: n_types = 16 constraints
    n_ic = n_types * n_types
    n_ir = n_types
    n_constraints = n_ic + n_ir
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # IC constraints
    row = 0
    for i in range(n_types):
        for i_prime in range(n_types):
            if i == i_prime:
                continue
            # -sum_j V[i,j]*x[i,j] + sum_j V[i,j]*x[i',j] + p[i] - p[i'] <= 0
            for j in range(n_goods):
                A_ub[row, i * n_goods + j] = -V[i, j]
                A_ub[row, i_prime * n_goods + j] = V[i, j]
            A_ub[row, n_x + i] = 1.0
            A_ub[row, n_x + i_prime] = -1.0
            b_ub[row] = 0.0
            row += 1
    
    # IR constraints
    for i in range(n_types):
        for j in range(n_goods):
            A_ub[row, i * n_goods + j] = -V[i, j]
        A_ub[row, n_x + i] = 1.0
        b_ub[row] = 0.0
        row += 1
    
    # Bounds: x[i,j] in [0,1], p[i] in [0, inf]
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen, but safety check
        pass
    
    if not result.success:
        # Fallback: simple VCG-like mechanism
        # Allocate each good to highest-value bidder, charge second-highest
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for j in range(n_goods):
            # Sort types by value for good j
            order = np.argsort(-V[:, j])
            # Allocate to top type
            allocations[order[0], j] = 1.0
            # Payment is second-highest value
            if len(order) > 1:
                payments[order[0]] += V[order[1], j]
        
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x = result.x[:n_x].reshape(n_types, n_goods)
    p = result.x[n_x:]
    
    # Clamp to ensure feasibility
    x = np.clip(x, 0, 1)
    p = np.clip(p, 0, None)
    
    # Verify constraints and fix if needed
    # Check IC
    for i in range(n_types):
        for i_prime in range(n_types):
            if i == i_prime:
                continue
            u_ii = np.dot(V[i, :], x[i, :]) - p[i]
            u_iip = np.dot(V[i, :], x[i_prime, :]) - p[i_prime]
            if u_ii < u_iip - regret_tol:
                # Adjust payment to fix
                p[i_prime] = min(p[i_prime], np.dot(V[i, :], x[i, :]) - u_iip + p[i] - regret_tol)
                p[i_prime] = max(0, p[i_prime])
    
    # Check IR
    for i in range(n_types):
        u_ii = np.dot(V[i, :], x[i, :]) - p[i]
        if u_ii < -regret_tol:
            p[i] = max(0, np.dot(V[i, :], x[i, :]))
    
    # Ensure payments are non-negative
    p = np.clip(p, 0, None)
    
    return {'allocations': x, 'payments': p}
# EVOLVE-BLOCK-END
