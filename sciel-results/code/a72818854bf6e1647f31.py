import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    # Demands: (source, target, weight)
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
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute load contributions for each demand in CW and CCW
    # cw_loads[i] is a vector of 10 ints
    # ccw_loads[i] is a vector of 10 ints
    
    cw_loads = np.zeros((n_demands, n_nodes), dtype=np.int64)
    ccw_loads = np.zeros((n_demands, n_nodes), dtype=np.int64)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        if s < t:
            cw_edges = range(s, t)
        elif s > t:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        else:
            cw_edges = []
            
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        for e in cw_edges:
            cw_loads[i, e] = w
            
        for e in ccw_edges:
            ccw_loads[i, e] = w
            
    # Formulate MILP
    # Variables:
    # x[0..n_demands-1]: binary, 1 if CCW, 0 if CW
    # x[n_demands]: continuous, T (max load)
    
    n_vars = n_demands + 1
    T_idx = n_demands
    
    # Objective: minimize T
    c = np.zeros(n_vars)
    c[T_idx] = 1.0
    
    # Integrality: x[0..n_demands-1] are integers, T is continuous
    integrality = np.zeros(n_vars)
    integrality[:n_demands] = 1.0
    
    # Bounds
    # x_i in [0, 1]
    # T in [0, infinity]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[T_idx] = np.inf
    
    # Constraints:
    # For each edge e:
    # load_e = sum_i (w_i * (1 - x_i) * A_i^cw[e] + w_i * x_i * A_i^ccw[e]) <= T
    # load_e = sum_i w_i * A_i^cw[e] + sum_i w_i * (A_i^ccw[e] - A_i^cw[e]) * x_i <= T
    # sum_i delta_{i,e} * x_i - T <= - base_e
    
    # Where:
    # base_e = sum_i w_i * A_i^cw[e]
    # delta_{i,e} = w_i * (A_i^ccw[e] - A_i^cw[e])
    
    base_load = np.sum(cw_loads, axis=0)
    deltas = ccw_loads - cw_loads # shape (n_demands, n_nodes)
    
    # Constraint matrix A shape (n_nodes, n_vars)
    A = np.zeros((n_nodes, n_vars))
    for e in range(n_nodes):
        A[e, :n_demands] = deltas[:, e]
        A[e, T_idx] = -1.0
        
    # Lower bound for constraints: -infinity
    # Upper bound for constraints: -base_load[e]
    cons_lb = -np.inf * np.ones(n_nodes)
    cons_ub = -base_load
    
    constraints = LinearConstraint(A, cons_lb, cons_ub)
    bounds = Bounds(lb, ub)
    
    start_time = time.time()
    timeout = 5.0 # seconds
    
    result = None
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
    except Exception:
        result = None
        
    if result is not None and result.status == 0 and result.success:
        x_sol = result.x
        # x_sol[0:n_demands] are the routing decisions
        # 1.0 means CCW, 0.0 means CW
        routing = x_sol[:n_demands].copy()
        # Ensure strict binary values for safety, though solver should provide them
        routing = np.where(routing > 0.5, 1.0, 0.0)
        return routing
    else:
        # Fallback to exhaustive search if MILP fails
        # This ensures we don't return garbage
        masks = np.arange(1 << n_demands, dtype=np.uint32)
        bit_positions = np.arange(n_demands, dtype=np.uint32)
        M = (masks[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1
        loads_all = M @ deltas + base_load[np.newaxis, :]
        max_loads = np.max(loads_all, axis=1)
        best_idx = np.argmin(max_loads)
        best_mask = masks[best_idx]
        
        routing = np.zeros(n_demands)
        for i in range(n_demands):
            if (best_mask >> i) & 1:
                routing[i] = 1.0
            else:
                routing[i] = 0.0
        return routing
