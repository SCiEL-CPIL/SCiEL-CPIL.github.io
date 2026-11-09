# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem using Integer Linear Programming.
    Minimizes the maximum edge load.
    """
    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    d = len(demands)
    
    # Pre-compute clockwise and counter-clockwise edge usage matrices
    # cw[i, e] = 1 if demand i uses edge e when routed clockwise
    cw = np.zeros((d, n), dtype=np.int64)
    ccw = np.zeros((d, n), dtype=np.int64)
    w = np.zeros(d, dtype=np.int64)
    
    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            # Self-loop, doesn't add load to edges (or depends on definition, 
            # but usually s!=t in these problems. If s==t, both paths are empty? 
            # Problem says "Clockwise from s to t uses edges s...t-1". If s==t, empty set.
            # So load is 0.
            continue
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % n
        ccw[i] = 1 - cw[i]
        
    # Variables:
    # x[i] for i in 0..d-1: 1 if demand i is routed counter-clockwise, 0 if clockwise
    # M: maximum load
    # Total variables = d + 1
    
    # Objective: minimize M
    # c = [0, 0, ..., 0, 1]
    c = np.zeros(d + 1)
    c[-1] = 1.0
    
    # Constraints:
    # For each edge e in 0..n-1:
    # Load_e = sum_i ( w_i * (x_i * ccw[i,e] + (1-x_i) * cw[i,e]) ) <= M
    # Load_e = sum_i ( w_i * cw[i,e] ) + sum_i ( w_i * (ccw[i,e] - cw[i,e]) * x_i ) <= M
    # Let base_load_e = sum_i w_i * cw[i,e]
    # Let delta_e_i = w_i * (ccw[i,e] - cw[i,e])
    # base_load_e + sum_i delta_e_i * x_i - M <= 0
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    for e in range(n):
        base_load = np.dot(w, cw[:, e])
        delta = w * (ccw[:, e] - cw[:, e])
        
        # Row: [delta_0, delta_1, ..., delta_{d-1}, -1]
        row = np.zeros(d + 1)
        row[:d] = delta
        row[-1] = -1.0
        
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(base_load) # base_load + delta @ x - M <= 0  => delta @ x - M <= -base_load ? No.
        
        # Wait: Load_e <= M
        # base_load_e + delta @ x <= M
        # delta @ x - M <= -base_load_e
        # So upper bound is -base_load_e
        
        b_upper[-1] = -base_load
    
    # Also need M >= 0 (implied by loads being non-negative, but good to be explicit if needed, 
    # though M is free variable in milp unless bounded. Loads are non-negative, so M>=0 is natural).
    # We can add a constraint M >= 0 if necessary, but usually the load constraints enforce it.
    
    A = np.array(A_rows)
    lb = np.array(b_lower)
    ub = np.array(b_upper)
    
    # Variable bounds
    # x[i] in {0, 1}
    # M in [0, inf)
    integrality = np.ones(d + 1, dtype=np.int8)
    integrality[-1] = 0  # M is continuous
    
    lb_vars = np.zeros(d + 1)
    ub_vars = np.ones(d + 1)
    ub_vars[-1] = np.inf
    
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(lb_vars, ub_vars)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:
            x = res.x[:d]
            # x should be 0 or 1
            routing = np.round(x).astype(np.float64)
            return routing
        else:
            # Fallback if MILP fails (unlikely for this small problem)
            # Return all clockwise (0)
            return np.zeros(d, dtype=np.float64)
            
    except Exception:
        return np.zeros(d, dtype=np.float64)

# EVOLVE-BLOCK-END
