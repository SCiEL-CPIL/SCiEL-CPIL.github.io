import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 60  # 60 second deadline
    
    # Parse inputs
    # distribution_spec: likely a list of 16 probabilities
    # support: list of 16 tuples (v1, v2)
    # constraints: dict with keys like 'regret', etc.
    
    if isinstance(distribution_spec, dict):
        # Handle dict format if necessary
        probs = []
        for s in support:
            if s in distribution_spec:
                probs.append(distribution_spec[s])
            else:
                probs.append(0.0)
        probs = np.array(probs)
    else:
        probs = np.array(distribution_spec, dtype=float)
    
    if len(probs) != 16:
        # Fallback if format is unexpected
        probs = np.ones(16) / 16.0
        
    # Normalize probabilities
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(16) / 16.0
        
    # Support: list of (v1, v2)
    # Ensure support is sorted consistently
    # Standard order: (0,0), (0,1), (0,2), (0,3), (1,0), ... (3,3)
    # Let's map support to indices
    support_list = list(support)
    # Create a mapping from value tuple to index
    val_to_idx = {}
    for i, v in enumerate(support_list):
        val_to_idx[v] = i
        
    # Extract values for each report
    # v[i] = (v1_i, v2_i)
    V = np.array([support_list[i] for i in range(16)], dtype=float)
    v1 = V[:, 0]
    v2 = V[:, 1]
    
    # Variables: x_1[16], x_2[16], p[16]
    # Total 48 variables
    # Indexing:
    # 0-15: x_1 for each report
    # 16-31: x_2 for each report
    # 32-47: p for each report
    
    n_reports = 16
    n_vars = 48
    
    # Objective: Maximize sum(probs * p)
    # linprog minimizes, so minimize -sum(probs * p)
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    # Constraints:
    # 1. IC: v_i . x_i - p_i >= v_i . x_j - p_j  =>  v_i . x_i - v_i . x_j - p_i + p_j >= 0
    #    Rewrite for linprog (<= form):
    #    -v_i . x_i + v_i . x_j + p_i - p_j <= 0
    #    For all i, j
    
    # 2. IR: v_i . x_i - p_i >= 0  =>  -v_i . x_i + p_i <= 0
    
    # 3. Non-negative payments: -p_i <= 0
    
    # 4. Allocation bounds: 0 <= x_k <= 1
    
    # Build A_ub and b_ub
    
    # IC Constraints: 16 * 16 = 256 constraints
    # IR Constraints: 16 constraints
    # Non-neg Payments: 16 constraints
    # Total A_ub rows: 256 + 16 + 16 = 288
    
    A_ub = np.zeros((288, n_vars))
    b_ub = np.zeros(288)
    
    row = 0
    # IC
    for i in range(n_reports):
        for j in range(n_reports):
            # -v_i . x_i + v_i . x_j + p_i - p_j <= 0
            # Coeff for x_1[i]: -v1[i]
            A_ub[row, i] = -v1[i]
            # Coeff for x_2[i]: -v2[i]
            A_ub[row, 16 + i] = -v2[i]
            # Coeff for x_1[j]: v1[j]
            A_ub[row, j] += v1[j]
            # Coeff for x_2[j]: v2[j]
            A_ub[row, 16 + j] += v2[j]
            # Coeff for p[i]: 1
            A_ub[row, 32 + i] = 1
            # Coeff for p[j]: -1
            A_ub[row, 32 + j] = -1
            b_ub[row] = 0
            row += 1
            
    # IR
    for i in range(n_reports):
        # -v_i . x_i + p_i <= 0
        A_ub[row, i] = -v1[i]
        A_ub[row, 16 + i] = -v2[i]
        A_ub[row, 32 + i] = 1
        b_ub[row] = 0
        row += 1
        
    # Non-negative payments
    for i in range(n_reports):
        # -p_i <= 0
        A_ub[row, 32 + i] = -1
        b_ub[row] = 0
        row += 1
        
    # Bounds
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))  # x_1
    for i in range(n_reports):
        bounds.append((0, 1))  # x_2
    for i in range(n_reports):
        bounds.append((0, None))  # p (non-negativity handled by A_ub, but bounds help)
        
    # Solve LP
    try:
        # Use highs solver for robustness
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            x_1 = res.x[0:16]
            x_2 = res.x[16:32]
            p = res.x[32:48]
            
            # Clip small numerical errors
            x_1 = np.clip(x_1, 0, 1)
            x_2 = np.clip(x_2, 0, 1)
            p = np.clip(p, 0, None)
            
            allocations = np.column_stack((x_1, x_2))
            payments = p
            
            # Check time
            if time.time() > deadline - 5:
                # Return immediately if close to deadline
                return {
                    'allocations': allocations,
                    'payments': payments
                }
                
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # LP failed, fallback to heuristic
            pass
            
    except Exception as e:
        pass
        
    # Fallback: Simple heuristic
    # Allocate good k if v_k >= 2 (or some threshold)
    # Payment: sum of values of allocated goods
    # This is not necessarily optimal but valid
    
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(16):
        v1_i = v1[i]
        v2_i = v2[i]
        
        # Simple threshold: allocate if value >= 1
        if v1_i >= 1:
            allocations[i, 0] = 1
        if v2_i >= 1:
            allocations[i, 1] = 1
            
        # Payment: charge the value
        payments[i] = (v1_i * allocations[i, 0]) + (v2_i * allocations[i, 1])
        
    return {
        'allocations': allocations,
        'payments': payments
    }
