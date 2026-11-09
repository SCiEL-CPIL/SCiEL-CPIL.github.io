# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute CW edge incidence matrix (D x N)
    # A[i, e] = 1 if edge e is on the clockwise path of demand i
    A_cw = np.zeros((D, N), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            for e in range(s, t):
                A_cw[i, e] = 1
        else:
            for e in range(s, N):
                A_cw[i, e] = 1
            for e in range(0, t):
                A_cw[i, e] = 1
    
    # CCW incidence is complement
    A_ccw = 1.0 - A_cw
    
    weights = np.array([w for s, t, w in demands], dtype=np.float64)
    
    # Min-max LP:
    # Variables: x_0..x_{D-1} (binary direction), t (max load)
    # Minimize: t
    # Subject to: for each edge e:
    #   sum_i (A_cw[i,e] * (1 - x_i) + A_ccw[i,e] * x_i) * w_i <= t
    #   => sum_i (A_cw[i,e] * w_i) + sum_i ((A_ccw[i,e] - A_cw[i,e]) * w_i) * x_i <= t
    #   => -sum_i ((A_ccw[i,e] - A_cw[i,e]) * w_i) * x_i + t >= sum_i (A_cw[i,e] * w_i)
    #   => sum_i ((A_ccw[i,e] - A_cw[i,e]) * w_i) * x_i - t <= -sum_i (A_cw[i,e] * w_i)
    
    # Build constraint matrix for linprog (minimize c @ x subject to A_ub @ x <= b_ub)
    # Variables: [x_0, ..., x_{D-1}, t]
    # Objective: minimize t -> c = [0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t
    
    # Constraints: for each edge e
    A_ub = np.zeros((N, D + 1))
    b_ub = np.zeros(N)
    for e in range(N):
        # sum_i (A_ccw[i,e] - A_cw[i,e]) * w_i * x_i - t <= -sum_i A_cw[i,e] * w_i
        for i in range(D):
            A_ub[e, i] = (A_ccw[i, e] - A_cw[i, e]) * weights[i]
        A_ub[e, D] = -1.0
        b_ub[e] = -np.sum(A_cw[:, e] * weights)
    
    # Bounds: x_i in [0, 1], t >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Solve with HiGHS
    start_time = time.time()
    deadline = start_time + 600.0
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs', options={'presolve': True, 'time_limit': 580})
        
        if result.success:
            routing = result.x[:D]
            # Round to nearest 0/1
            routing = (routing >= 0.5).astype(np.float64)
            
            # Verify and compute actual max load
            loads = np.zeros(N, dtype=np.float64)
            for i in range(D):
                if routing[i] < 0.5:
                    loads += A_cw[i] * weights[i]
                else:
                    loads += A_ccw[i] * weights[i]
            max_load = np.max(loads)
        else:
            # Fallback: all clockwise
            routing = np.zeros(D)
            loads = A_cw.T @ weights
            max_load = np.max(loads)
    except Exception:
        # Fallback: all clockwise
        routing = np.zeros(D)
        loads = A_cw.T @ weights
        max_load = np.max(loads)
    
    # Local search polish from the LP solution
    if time.time() < deadline - 10.0:
        def compute_max_load(r):
            loads = np.zeros(N, dtype=np.float64)
            for i in range(D):
                if r[i] < 0.5:
                    loads += A_cw[i] * weights[i]
                else:
                    loads += A_ccw[i] * weights[i]
            return np.max(loads)
        
        best_routing = routing.copy()
        best_load = max_load
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(D):
                if time.time() > deadline - 10.0:
                    break
                trial = best_routing.copy()
                trial[i] = 1.0 - trial[i]
                load = compute_max_load(trial)
                if load < best_load - 1e-9:
                    best_routing = trial
                    best_load = load
                    improved = True
    
    return best_routing

# EVOLVE-BLOCK-END
