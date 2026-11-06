# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s limit
    
    # Parse inputs
    # support: list of 16 types, each type is (v1, v2)
    # distribution_spec: dict or list of probabilities for the 16 types
    # constraints: may contain additional info
    
    # Extract probabilities
    if isinstance(distribution_spec, dict):
        # Try to get probabilities from dict
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            # Assume keys are type indices or values
            probs = np.array([distribution_spec.get(i, 0.0) for i in range(16)], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(16) / 16.0
    
    # Extract support (list of 16 types, each (v1, v2))
    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i // 4, i % 4) for i in range(16)]
    
    # Ensure we have 16 types
    if len(support_list) != 16:
        support_list = [(i // 4, i % 4) for i in range(16)]
    
    # Build value matrix: V[i, j] = value of good j for type i
    V = np.zeros((16, 2))
    for i, (v1, v2) in enumerate(support_list):
        V[i, 0] = v1
        V[i, 1] = v2
    
    # Variables: x_{i,0}, x_{i,1}, p_i for i=0..15
    # Total variables: 16*2 + 16 = 48
    # Indexing: x_{i,j} at index 2*i + j, p_i at index 32 + i
    
    n_vars = 48
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for i in range(16):
        c[32 + i] = -probs[i]
    
    # Constraints:
    # 1. IR: for each i, sum_j V[i,j]*x_{i,j} - p_i >= 0
    #    => sum_j V[i,j]*x_{i,j} - p_i >= 0
    #    In standard form: -sum_j V[i,j]*x_{i,j} + p_i <= 0
    A_ir = np.zeros((16, n_vars))
    b_ir = np.zeros(16)
    for i in range(16):
        A_ir[i, 2*i] = -V[i, 0]
        A_ir[i, 2*i+1] = -V[i, 1]
        A_ir[i, 32+i] = 1
    
    # 2. IC: for all i, i', sum_j V[i,j]*x_{i',j} - p_{i'} <= sum_j V[i,j]*x_{i,j} - p_i
    #    => sum_j V[i,j]*x_{i',j} - p_{i'} - sum_j V[i,j]*x_{i,j} + p_i <= 0
    n_ic = 16 * 16
    A_ic = np.zeros((n_ic, n_vars))
    b_ic = np.zeros(n_ic)
    
    for i in range(16):
        for ip in range(16):
            idx = i * 16 + ip
            # Coefficients for x_{ip,j}: V[i,j]
            A_ic[idx, 2*ip] += V[i, 0]
            A_ic[idx, 2*ip+1] += V[i, 1]
            # Coefficient for p_{ip}: -1
            A_ic[idx, 32+ip] += -1
            # Coefficients for x_{i,j}: -V[i,j]
            A_ic[idx, 2*i] += -V[i, 0]
            A_ic[idx, 2*i+1] += -V[i, 1]
            # Coefficient for p_i: +1
            A_ic[idx, 32+i] += 1
    
    # Combine constraints
    A = np.vstack([A_ir, A_ic])
    b = np.concatenate([b_ir, b_ic])
    
    # Bounds: x_{i,j} in [0, 1], p_i in [0, inf)
    bounds = []
    for i in range(16):
        bounds.append((0, 1))  # x_{i,0}
        bounds.append((0, 1))  # x_{i,1}
    for i in range(16):
        bounds.append((0, None))  # p_i
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a simple valid mechanism
            # Award both goods to all types, charge 0
            allocations = np.ones((16, 2))
            payments = np.zeros(16)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[2*i]
            allocations[i, 1] = x[2*i+1]
            payments[i] = x[32+i]
        
        # Verify constraints with tolerance
        # Check IR
        for i in range(16):
            util = np.dot(V[i], allocations[i]) - payments[i]
            if util < -1e-8:
                # Fix: set payment to max(0, dot(V[i], allocations[i]))
                payments[i] = max(0.0, np.dot(V[i], allocations[i]))
        
        # Check IC
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                if util_report - util_truth > 1e-8:
                    # This violates IC; need to fix
                    # Adjust payments to ensure IC
                    pass
        
        # Re-verify and fix if needed
        # Simple fix: if any IC violation, use a conservative mechanism
        ic_ok = True
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                if util_report - util_truth > 1e-8:
                    ic_ok = False
                    break
            if not ic_ok:
                break
        
        if not ic_ok:
            # Fallback: award nothing, charge 0
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception:
        # Fallback: return a simple valid mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
