"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    """
    Construct a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A (unordered pairs, equal summands allowed).
    Minimize the largest representation multiplicity.
    
    Approach:
    Formulate as an Integer Linear Programming (ILP) problem.
    Variables:
    x_i (binary) for i in [0, 128] indicating if i is in A.
    y_{a,b} (binary) for pairs a <= b indicating if both a and b are in A.
    M (integer) representing the max multiplicity.
    
    Objective: Minimize M.
    Constraints:
    1. Coverage: For each s in [128, 256], sum of y_{a,b} where a+b=s >= 1.
    2. Multiplicity: For each s in [128, 256], sum of y_{a,b} where a+b=s <= M.
    3. Linearization: y_{a,b} <= x_a, y_{a,b} <= x_b, y_{a,b} >= x_a + x_b - 1.
    """
    n = 128
    target_low = 128
    target_high = 256
    targets = list(range(target_low, target_high + 1))
    num_targets = len(targets)
    
    # Generate valid pairs (a, b) with a <= b and target_low <= a + b <= target_high
    pairs = []
    pair_sums = []
    for a in range(n + 1):
        for b in range(a, n + 1):
            s = a + b
            if target_low <= s <= target_high:
                pairs.append((a, b))
                pair_sums.append(s)
    
    num_pairs = len(pairs)
    
    # Variable indices:
    # x_i for i in [0, n] : 0 to n
    # y_k for k in [0, num_pairs) : n+1 to n+num_pairs
    # M : n+num_pairs+1
    N_x = n + 1
    N_y = num_pairs
    N_M = 1
    N_total = N_x + N_y + N_M
    
    idx_M = N_x + N_y
    
    # Objective: Minimize M
    c = np.zeros(N_total)
    c[idx_M] = 1.0
    
    # Bounds
    lb = np.zeros(N_total)
    ub = np.ones(N_total)
    # M can be larger
    ub[idx_M] = N_x # Safe upper bound
    
    integrality = np.zeros(N_total)
    integrality[:N_x] = 1
    integrality[N_x:N_x+N_y] = 1
    integrality[idx_M] = 1 # Force integer M
    
    # Constraints
    # Total constraints: num_targets (coverage) + num_targets (upper bound) + 3 * num_pairs (linearization)
    num_constraints = num_targets + num_targets + 3 * num_pairs
    
    A_all = np.zeros((num_constraints, N_total))
    lb_all = np.zeros(num_constraints)
    ub_all = np.zeros(num_constraints)
    
    # 1. Coverage: sum(y_k for pairs summing to s) >= 1
    for k, s in enumerate(pair_sums):
        row_idx = s - target_low
        A_all[row_idx, N_x + k] = 1.0
        
    lb_all[:num_targets] = 1.0
    ub_all[:num_targets] = np.inf
    
    # 2. Upper bound on multiplicity: sum(y_k) - M <= 0
    for k, s in enumerate(pair_sums):
        row_idx = num_targets + (s - target_low)
        A_all[row_idx, N_x + k] = 1.0
        A_all[row_idx, idx_M] = -1.0
        
    lb_all[num_targets:2*num_targets] = -np.inf
    ub_all[num_targets:2*num_targets] = 0.0
    
    # 3. Linearization constraints
    start_idx = 2 * num_targets
    for k, (a, b) in enumerate(pairs):
        y_idx = N_x + k
        x_a_idx = a
        x_b_idx = b
        
        # y_k - x_a <= 0
        A_all[start_idx + 3*k, y_idx] = 1.0
        A_all[start_idx + 3*k, x_a_idx] = -1.0
        lb_all[start_idx + 3*k] = -np.inf
        ub_all[start_idx + 3*k] = 0.0
        
        # y_k - x_b <= 0
        A_all[start_idx + 3*k + 1, y_idx] = 1.0
        A_all[start_idx + 3*k + 1, x_b_idx] = -1.0
        lb_all[start_idx + 3*k + 1] = -np.inf
        ub_all[start_idx + 3*k + 1] = 0.0
        
        # x_a + x_b - y_k <= 1
        A_all[start_idx + 3*k + 2, x_a_idx] = 1.0
        A_all[start_idx + 3*k + 2, x_b_idx] = 1.0
        A_all[start_idx + 3*k + 2, y_idx] = -1.0
        lb_all[start_idx + 3*k + 2] = -np.inf
        ub_all[start_idx + 3*k + 2] = 1.0
    
    # Solve ILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A_all, lb_all, ub_all),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={'time_limit': 50}
        )
        
        if res.status == 0:
            x_sol = res.x[:N_x]
            A_set = {i for i in range(N_x) if x_sol[i] > 0.5}
            
            # Verify coverage and calculate score
            # If valid, return it
            # Simple check: ensure all targets covered
            covered = [False] * num_targets
            A_list = sorted(A_set)
            for i in range(len(A_list)):
                for j in range(i, len(A_list)):
                    s = A_list[i] + A_list[j]
                    if target_low <= s <= target_high:
                        covered[s - target_low] = True
            
            if all(covered):
                return A_list
        
    except Exception:
        pass
        
    # Fallback: Return a known valid set (seed)
    return list(range(64, 129))
# EVOLVE-BLOCK-END
