"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)

def _sa_solve(a, b, time_limit):
    """
    Solve QAP using Simulated Annealing.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize with a random permutation
    p = np.random.permutation(n).astype(np.int32)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    T = 1000.0
    alpha = 0.999
    
    while time.time() < deadline:
        # Generate random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        li = p[i]
        lj = p[j]
        
        # Calculate delta
        diff_row = b[lj] - b[li]
        diff_col = b[:, lj] - b[:, li]
        
        dr_p = diff_row[p]
        dc_p = diff_col[p]
        
        t1 = np.dot(a[i], dr_p)
        t2 = np.dot(a[:, i], dc_p)
        t3 = -np.dot(a[j], dr_p)
        t4 = -np.dot(a[:, j], dc_p)
        
        total_cross = t1 + t2 + t3 + t4
        
        # Subtract k=i and k=j terms
        c_ii = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li]) + \
               (a[i, i] - a[i, j]) * (b[li, lj] - b[li, li])
               
        c_jj = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj]) + \
               (a[j, i] - a[j, j]) * (b[lj, lj] - b[lj, li])
               
        cross_sum_excl = total_cross - c_ii - c_jj
        
        d_self = a[i, i] * (b[lj, lj] - b[li, li]) + \
                 a[j, j] * (b[li, li] - b[lj, lj]) + \
                 a[i, j] * (b[lj, li] - b[li, lj]) + \
                 a[j, i] * (b[li, lj] - b[lj, li])
                 
        delta = d_self + cross_sum_excl
        
        # Acceptance
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        T *= alpha
        if T < 1e-4:
            T = 1e-4
            
    return best_p, best_cost

def solve(a, b):
    """
    Solve the QAP instance using Simulated Annealing.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    p, cost = _sa_solve(a, b, deadline - start_time)
    
    return p.tolist()
# EVOLVE-BLOCK-END
