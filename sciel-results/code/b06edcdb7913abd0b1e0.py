"""Incentive-compatible finite auction design via direct LP solve."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Parse support: 16 types (v1, v2) where v1, v2 in {0,1,2,3}
    # Standard ordering: v1 * 4 + v2
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    else:
        support = list(support)
    
    n_types = len(support)
    if n_types != 16:
        # Try to construct standard support
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        n_types = 16
    
    # Parse probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    if len(probs) != 16:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(16) / 16.0
    
    # Ensure non-negative
    probs = np.maximum(probs, 0)
    
    # Values: type i has value (v1_i, v2_i)
    values = np.array(support, dtype=float)  # shape (16, 2)
    
    # Allocation: for each type i, x_i = (x_i1, x_i2) where x_ij in [0, 1]
    # Payment: for each type i, p_i >= 0
    
    # Variables: x_01, x_02, ..., x_151, x_152, p_0, p_1, ..., p_15
    # Total: 32 + 16 = 48 variables
    # But we can eliminate payments using IC constraints.
    
    # For a single additive bidder with 2 goods, the IC condition is:
    # u_i = v_i · x_i - p_i
    # u_i >= u_j for all j (truthfulness)
    # p_i >= 0 (non-negative payments)
    # u_i >= 0 (individual rationality)
    
    # The payment can be expressed as:
    # p_i = v_i · x_i - u_i
    # where u_i is the utility.
    
    # IC: u_i - v_i · x_i >= u_j - v_j · x_j for all i, j
    # IR: u_i - v_i · x_i >= 0 for all i
    # p_i = v_i · x_i - u_i >= 0 => u_i <= v_i · x_i
    
    # Objective: maximize E[p] = sum_i probs_i * (v_i · x_i - u_i)
    #           = sum_i probs_i * v_i · x_i - sum_i probs_i * u_i
    
    # Variables: x_0..x_31 (allocations) and u_0..u_15 (utilities)
    # Total: 32 + 16 = 48 variables
    
    n_x = 32  # 16 types * 2 goods
    n_u = 16  # 16 utilities
    n_vars = n_x + n_u
    
    # Objective: maximize sum_i probs_i * (v_i1 * x_i1 + v_i2 * x_i2 - u_i)
    # c = [probs_i * v_i1, probs_i * v_i2, -probs_i] for each i
    c = np.zeros(n_vars)
    for i in range(16):
        c[i * 2] = probs[i] * values[i, 0]
        c[i * 2 + 1] = probs[i] * values[i, 1]
        c[n_x + i] = -probs[i]
    
    # linprog minimizes, so negate
    c_min = -c
    
    # Constraints:
    # 1. IC: for all i, j: (u_i - v_i · x_i) - (u_j - v_j · x_j) >= 0
    #    => -v_i1*x_i1 - v_i2*x_i2 + u_i + v_j1*x_j1 + v_j2*x_j2 - u_j <= 0
    # 2. IR: u_i - v_i · x_i >= 0 => -v_i1*x_i1 - v_i2*x_i2 + u_i <= 0
    # 3. Non-negativity of allocations: x_ij >= 0 (handled by bounds)
    # 4. x_ij <= 1 (handled by bounds)
    # 5. u_i <= v_i · x_i (from p_i >= 0): u_i - v_i1*x_i1 - v_i2*x_i2 <= 0
    
    # Let's build A_ub and b_ub for inequality constraints A_ub @ z <= b_ub
    
    A_rows = []
    b_vals = []
    
    # IC constraints: 16 * 16 = 256 constraints
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -v_i1*x_i1 - v_i2*x_i2 + u_i + v_j1*x_j1 + v_j2*x_j2 - u_j <= 0
            row[i * 2] = -values[i, 0]
            row[i * 2 + 1] = -values[i, 1]
            row[n_x + i] = 1
            row[j * 2] = values[j, 0]
            row[j * 2 + 1] = values[j, 1]
            row[n_x + j] = -1
            A_rows.append(row)
            b_vals.append(0)
    
    # IR constraints: 16 constraints
    # u_i - v_i1*x_i1 - v_i2*x_i2 >= 0 => -u_i + v_i1*x_i1 + v_i2*x_i2 <= 0
    for i in range(16):
        row = np.zeros(n_vars)
        row[i * 2] = values[i, 0]
        row[i * 2 + 1] = values[i, 1]
        row[n_x + i] = -1
        A_rows.append(row)
        b_vals.append(0)
    
    # p_i >= 0 => v_i · x_i - u_i >= 0 => u_i - v_i1*x_i1 - v_i2*x_i2 <= 0
    for i in range(16):
        row = np.zeros(n_vars)
        row[i * 2] = -values[i, 0]
        row[i * 2 + 1] = -values[i, 1]
        row[n_x + i] = 1
        A_rows.append(row)
        b_vals.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: x_ij in [0, 1], u_i in [0, infinity)
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
    
    # Check time
    if time.time() > deadline - 10:
        # Return a simple feasible solution
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    # Solve LP
    try:
        result = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback: return zero allocation
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if time.time() > deadline - 5:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result.success:
        x_sol = result.x[:n_x]
        u_sol = result.x[n_x:n_x + n_u]
        
        # Extract allocations
        allocations = x_sol.reshape(16, 2)
        
        # Compute payments: p_i = v_i · x_i - u_i
        payments = np.zeros(16)
        for i in range(16):
            payments[i] = values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1] - u_sol[i]
        
        # Ensure non-negative (clip small negative values)
        payments = np.maximum(payments, 0)
        
        # Verify IC and IR with tolerance
        # Check IC: u_i >= u_j + (v_i - v_j) · x_i ... actually let's verify directly
        # u_i = v_i · x_i - p_i
        utils = values @ allocations.T - payments  # shape (16,)
        
        # Check IC: for all i, j: utils[i] >= utils[j] + (v_i - v_j) · x_i
        # Actually, the IC condition is: v_i · x_i - p_i >= v_i · x_j - p_j for all i, j
        # i.e., utils[i] >= v_i · x_j - p_j
        
        # Let's just verify the solution is feasible
        max_violation = 0
        for i in range(16):
            for j in range(16):
                lhs = utils[i]
                rhs = values[i, 0] * allocations[j, 0] + values[i, 1] * allocations[j, 1] - payments[j]
                violation = rhs - lhs
                if violation > max_violation:
                    max_violation = violation
        
        if max_violation > 1e-8:
            # Solution not feasible, try to fix or return zero
            # For now, return zero allocation
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
        else:
            return {'allocations': allocations, 'payments': payments}
    else:
        # LP failed, return zero allocation
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
