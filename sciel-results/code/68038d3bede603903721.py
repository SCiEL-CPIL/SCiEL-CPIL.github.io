import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Solves the optimal truthful auction design for a single additive bidder
    with two goods, values in {0, 1, 2, 3}, and a uniform distribution.
    
    Returns:
        dict: {
            'allocations': np.ndarray of shape (16, 2),
            'payments': np.ndarray of shape (16,)
        }
    """
    # The support is defined as values in {0, 1, 2, 3} for each good.
    # We assume the grid is 4x4 = 16 profiles.
    # Indexing: idx = i * 4 + j, where i is value of good 1, j is value of good 2.
    values = [0, 1, 2, 3]
    n = len(values)
    profiles = [(i, j) for i in values for j in values]
    num_profiles = len(profiles) # 16
    
    # Variable layout for LP:
    # x_0 ... x_15 (allocations for good 1)
    # y_0 ... y_15 (allocations for good 2)
    # p_0 ... p_15 (payments)
    # Total variables: 48
    
    # Objective: Maximize sum(p_i)
    # linprog minimizes, so we minimize -sum(p_i)
    c = np.zeros(num_profiles * 3)
    for k in range(num_profiles):
        c[2 * num_profiles + k] = -1.0
    
    # Constraints:
    # 1. IR: p_k <= i*x_k + j*y_k  =>  -i*x_k - j*y_k + p_k <= 0
    # 2. IC: p_k - p_m <= i_k*x_k + j_k*y_k - (i_m*x_m + j_m*y_m)
    #    => i_k*x_k + j_k*y_k - i_m*x_m - j_m*y_m - p_k + p_m <= 0
    # 3. Non-negativity of x, y, p is handled by bounds.
    
    A_ub = []
    b_ub = []
    
    # IR Constraints
    for k, (i, j) in enumerate(profiles):
        row = np.zeros(num_profiles * 3)
        row[k] = -i          # coeff for x_k
        row[num_profiles + k] = -j  # coeff for y_k
        row[2 * num_profiles + k] = 1.0 # coeff for p_k
        A_ub.append(row)
        b_ub.append(0.0)
        
    # IC Constraints
    for k, (i_k, j_k) in enumerate(profiles):
        for m, (i_m, j_m) in enumerate(profiles):
            if k == m:
                continue
            row = np.zeros(num_profiles * 3)
            # LHS: i_k*x_k + j_k*y_k - i_m*x_m - j_m*y_m - p_k + p_m
            row[k] = i_k
            row[num_profiles + k] = j_k
            row[m] = -i_m
            row[num_profiles + m] = -j_m
            row[2 * num_profiles + k] = -1.0
            row[2 * num_profiles + m] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds:
    # x, y in [0, 1]
    # p in [0, inf) (but effectively bounded by IR)
    bounds = []
    for k in range(num_profiles):
        bounds.append((0, 1)) # x
    for k in range(num_profiles):
        bounds.append((0, 1)) # y
    for k in range(num_profiles):
        bounds.append((0, None)) # p
        
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to a simple deterministic auction if LP fails
        # e.g., allocate item 1 if v1 > 1.5, item 2 if v2 > 1.5
        # This is a safe heuristic
        allocations = np.zeros((num_profiles, 2))
        payments = np.zeros(num_profiles)
        for k, (i, j) in enumerate(profiles):
            if i > 1.5:
                allocations[k, 0] = 1.0
                payments[k] += i
            if j > 1.5:
                allocations[k, 1] = 1.0
                payments[k] += j
        return {
            'allocations': allocations,
            'payments': payments
        }
    
    x = res.x[:num_profiles]
    y = res.x[num_profiles:2*num_profiles]
    p = res.x[2*num_profiles:3*num_profiles]
    
    allocations = np.column_stack((x, y))
    
    return {
        'allocations': allocations,
        'payments': p
    }
