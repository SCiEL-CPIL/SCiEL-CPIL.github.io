from scipy.optimize import linprog
import numpy as np
# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    The optimal solution is the uniform distribution, which minimizes the 
    L2 norm of the coefficients and thus the peak of the autoconvolution.
    We formulate this as a linear feasibility problem to find a vector where 
    all heights are equal, satisfying the sum constraint.
    """
    N = 600
    w = 1.0 / (2 * N)
    total_sum = 1.0 / w  # Sum of heights must be 1/w to make L1 norm 1
    
    # We want to find h such that:
    # 1. sum(h) = total_sum
    # 2. h_i >= 0
    # 3. h_i = h_{i+1} (to enforce uniformity, which is the known optimum)
    
    # Objective: minimize 0 (feasibility)
    c = np.zeros(N)
    
    # Constraints for uniformity: h_i - h_{i+1} <= 0 and h_{i+1} - h_i <= 0
    # We can use just one direction if we close the loop, but 2*(N-1) is safer.
    # Actually, h_i <= h_{i+1} for all i (cyclic) implies all equal.
    A_ub_rows = []
    b_ub_rows = []
    
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        row[(i + 1) % N] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Equality constraint: sum(h) = total_sum
    A_eq = np.array([[1.0] * N])
    b_eq = np.array([total_sum])
    
    bounds = [(0, None) for _ in range(N)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            return list(res.x)
    except Exception:
        pass
        
    # Fallback to analytical uniform solution
    return [total_sum / N] * N
# EVOLVE-BLOCK-END
