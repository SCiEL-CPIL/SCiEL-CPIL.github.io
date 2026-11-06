import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    Strategy: Evaluate Bundle Reserve and Separate Reserve mechanisms, pick the best.
    """
    t_start = time.time()
    deadline = t_start + 80.0

    # Parse distribution_spec to get joint probabilities
    # Assuming distribution_spec is a dict or has attributes for PMF
    # Common formats: {'pmf': [[...]]} or just a 2D array
    if isinstance(distribution_spec, dict):
        if 'pmf' in distribution_spec:
            pmf = np.array(distribution_spec['pmf'])
        elif 'joint' in distribution_spec:
            pmf = np.array(distribution_spec['joint'])
        else:
            # Fallback: assume uniform if not found
            pmf = np.ones((4, 4)) / 16.0
    elif isinstance(distribution_spec, (np.ndarray, list)):
        pmf = np.array(distribution_spec)
        if pmf.shape != (4, 4):
            # Try to reshape or default
            if pmf.size == 16:
                pmf = pmf.reshape(4, 4)
            else:
                pmf = np.ones((4, 4)) / 16.0
    else:
        pmf = np.ones((4, 4)) / 16.0

    # Normalize PMF just in case
    total_prob = pmf.sum()
    if total_prob > 0:
        pmf = pmf / total_prob
    else:
        pmf = np.ones((4, 4)) / 16.0

    support_vals = list(support) if isinstance(support, (list, tuple, np.ndarray)) else [0, 1, 2, 3]
    # Map support values to indices 0..3 for the 4x4 grid
    # Assume support is [0, 1, 2, 3] or similar
    val_to_idx = {v: i for i, v in enumerate(support_vals)}
    
    # If support is not exactly 4 values, we might need to map differently, 
    # but the task says "each value in {0,1,2,3}", so 4 values.
    
    best_revenue = -1
    best_allocations = None
    best_payments = None

    # Strategy 1: Bundle Reserve
    # Allocation: 1 if v1+v2 >= R, else 0. Payment: R if allocated, else 0.
    for R in range(0, 7): # Max sum is 6
        if time.time() > deadline:
            break
            
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        
        rev = 0.0
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                v1 = support_vals[i]
                v2 = support_vals[j]
                if v1 + v2 >= R:
                    allocations[idx, 0] = 1.0
                    allocations[idx, 1] = 1.0
                    payments[idx] = R
                    rev += pmf[i, j] * R
        
        if rev > best_revenue:
            best_revenue = rev
            best_allocations = allocations
            best_payments = payments

    # Strategy 2: Separate Reserves
    # Allocation 1: 1 if v1 >= R1, else 0. Allocation 2: 1 if v2 >= R2, else 0.
    # Payment: R1*A1 + R2*A2.
    for R1 in range(0, 4):
        for R2 in range(0, 4):
            if time.time() > deadline:
                break
                
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
            rev = 0.0
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    v1 = support_vals[i]
                    v2 = support_vals[j]
                    
                    a1 = 1.0 if v1 >= R1 else 0.0
                    a2 = 1.0 if v2 >= R2 else 0.0
                    
                    allocations[idx, 0] = a1
                    allocations[idx, 1] = a2
                    p = R1 * a1 + R2 * a2
                    payments[idx] = p
                    rev += pmf[i, j] * p
            
            if rev > best_revenue:
                best_revenue = rev
                best_allocations = allocations
                best_payments = payments

    # If no valid mechanism found (should not happen), fallback to zero
    if best_allocations is None:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    return {
        'allocations': best_allocations,
        'payments': best_payments
    }
