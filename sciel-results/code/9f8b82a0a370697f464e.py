import numpy as np
from scipy.optimize import linprog
import json

def construct_auction(distribution_spec, support, constraints):
    # Determine support
    if support is None or len(support) == 0:
        # Default support: {0, 1, 2, 3} x {0, 1, 2, 3}
        vals = [0, 1, 2, 3]
        support = [(v0, v1) for v0 in vals for v1 in vals]
    
    N = len(support)
    
    # Determine probabilities
    if distribution_spec is None:
        # Uniform distribution
        probs = np.ones(N) / N
    else:
        if isinstance(distribution_spec, dict):
            probs = np.array([distribution_spec.get(t, 0.0) for t in support])
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            if len(distribution_spec) == N:
                probs = np.array(distribution_spec)
            else:
                # Fallback to uniform if length mismatch
                probs = np.ones(N) / N
        else:
            probs = np.ones(N) / N
            
    # Normalize probabilities just in case
    sum_p = np.sum(probs)
    if sum_p > 0:
        probs = probs / sum_p
    else:
        probs = np.ones(N) / N

    # Variable indices
    # 0..N-1: U_v
    # N..2N-1: x_v0
    # 2N..3N-1: x_v1
    
    n_vars = 3 * N
    
    # Objective function coefficients (minimize -revenue)
    c = np.zeros(n_vars)
    for i in range(N):
        v0, v1 = support[i]
        c[i] = -probs[i] # -pi_v * U_v
        c[N + i] = -probs[i] * v0 # -pi_v * v0 * x_v0
        c[2*N + i] = -probs[i] * v1 # -pi_v * v1 * x_v1
        
    # Inequality constraints A_ub * x <= b_ub
    A_ub = []
    b_ub = []
    
    # 1. U_v >= 0  =>  -U_v <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[i] = -1
        A_ub.append(row)
        b_ub.append(0)
        
    # 2. p_v >= 0 => U_v <= v . x_v => U_v - v0*x_v0 - v1*x_v1 <= 0
    for i in range(N):
        v0, v1 = support[i]
        row = np.zeros(n_vars)
        row[i] = 1
        row[N + i] = -v0
        row[2*N + i] = -v1
        A_ub.append(row)
        b_ub.append(0)
        
    # 3. IC: U_v >= U_v' + (v - v') . x_v'
    # => -U_v + U_v' + (v0-v0')*x_v0' + (v1-v1')*x_v1' <= 0
    for i in range(N):
        for j in range(N):
            if i == j: continue
            v0_i, v1_i = support[i]
            v0_j, v1_j = support[j]
            row = np.zeros(n_vars)
            row[i] = -1
            row[j] = 1
            row[N + j] = v0_i - v0_j
            row[2*N + j] = v1_i - v1_j
            A_ub.append(row)
            b_ub.append(0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, None)) # U_v
    for i in range(N):
        bounds.append((0, 1)) # x_v0
    for i in range(N):
        bounds.append((0, 1)) # x_v1
        
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback: return zero allocations and zero payments
        allocations = np.zeros((N, 2))
        payments = np.zeros(N)
        return {
            "allocations": allocations,
            "payments": payments
        }
        
    x_opt = res.x
    
    # Extract results
    allocations = np.zeros((N, 2))
    payments = np.zeros(N)
    
    for i in range(N):
        U_i = x_opt[i]
        x_i0 = x_opt[N + i]
        x_i1 = x_opt[2*N + i]
        
        # Clean up small numerical errors
        x_i0 = np.clip(x_i0, 0, 1)
        x_i1 = np.clip(x_i1, 0, 1)
        U_i = max(0, U_i)
        
        allocations[i] = [x_i0, x_i1]
        
        v0, v1 = support[i]
        p_i = v0 * x_i0 + v1 * x_i1 - U_i
        p_i = max(0, p_i)
        payments[i] = p_i
        
    return {
        "allocations": allocations,
        "payments": payments
    }
