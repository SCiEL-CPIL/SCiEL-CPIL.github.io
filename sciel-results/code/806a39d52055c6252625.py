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
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one value per demand.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s limit

    n_nodes = 10
    D = 15

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

    # Precompute CW and CCW edge weight matrices
    # cw_weights[i, e] = weight of demand i on edge e if routed CW, else 0
    # ccw_weights[i, e] = weight of demand i on edge e if routed CCW, else 0
    cw_weights = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_weights = np.zeros((D, n_nodes), dtype=np.float64)

    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        node = s
        while node != t:
            cw_weights[i, node] = w
            node = (node + 1) % n_nodes
        
        # Counter-clockwise edges: all other edges
        ccw_weights[i, :] = w - cw_weights[i, :]

    # base_load[e] = sum of cw_weights[i, e] for all i
    # This is the load if all demands are routed CW.
    base_load = np.sum(cw_weights, axis=0)
    
    # delta_load[i, e] = ccw_weights[i, e] - cw_weights[i, e]
    # If demand i is routed CCW (x_i = 1), we add delta_load[i, e] to the load.
    # If demand i is routed CW (x_i = 0), we add 0.
    # Load[e] = base_load[e] + sum_{i} x_i * delta_load[i, e]
    delta_load = ccw_weights - cw_weights

    # Formulate as LP:
    # Minimize L
    # Subject to: for each edge e, base_load[e] + sum_i x_i * delta_load[i, e] <= L
    #             0 <= x_i <= 1 (binary, but LP relaxation)
    #
    # Variables: x_0, ..., x_{D-1}, L
    # Total variables: D + 1 = 16
    #
    # Objective: minimize L -> c = [0, 0, ..., 0, 1]
    #
    # Constraints (A_ub @ vars <= b_ub):
    # For each edge e: sum_i delta_load[i, e] * x_i - L <= -base_load[e]
    # 
    # So A_ub has shape (n_nodes, D+1)
    # b_ub has shape (n_nodes,)
    
    c = np.zeros(D + 1)
    c[-1] = 1.0  # minimize L
    
    A_ub = np.zeros((n_nodes, D + 1))
    b_ub = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        A_ub[e, :D] = delta_load[:, e]
        A_ub[e, D] = -1.0  # coefficient for L
        b_ub[e] = -base_load[e]
    
    # Bounds: 0 <= x_i <= 1, L >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Solve using HiGHS (default in recent scipy)
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() > deadline:
        # Fallback to brute force if LP takes too long
        best_max_load = float('inf')
        best_combo = 0
        for combo in range(1 << D):
            if time.time() > deadline:
                break
            current_load = base_load.copy()
            cc = combo
            for i in range(D):
                if cc & 1:
                    current_load += delta_load[i]
                cc >>= 1
            max_load = np.max(current_load)
            if max_load < best_max_load:
                best_max_load = max_load
                best_combo = combo
        routing = np.zeros(D)
        for i in range(D):
            if (best_combo >> i) & 1:
                routing[i] = 1.0
            else:
                routing[i] = 0.0
        return routing
    
    if not result.success:
        # If LP fails, fall back to brute force
        best_max_load = float('inf')
        best_combo = 0
        for combo in range(1 << D):
            if time.time() > deadline:
                break
            current_load = base_load.copy()
            cc = combo
            for i in range(D):
                if cc & 1:
                    current_load += delta_load[i]
                cc >>= 1
            max_load = np.max(current_load)
            if max_load < best_max_load:
                best_max_load = max_load
                best_combo = combo
        routing = np.zeros(D)
        for i in range(D):
            if (best_combo >> i) & 1:
                routing[i] = 1.0
            else:
                routing[i] = 0.0
        return routing
    
    # Extract routing decisions
    # The LP solution may have fractional x_i values. Round them.
    x = result.x[:D]
    routing = np.where(x >= 0.5, 1.0, 0.0)
    
    # Verify the rounded solution and potentially improve with local search
    def compute_max_load(routing_vec):
        loads = base_load.copy()
        for i in range(D):
            if routing_vec[i] >= 0.5:
                loads += delta_load[i]
        return np.max(loads)
    
    current_max = compute_max_load(routing)
    
    # Local search: try flipping each bit
    improved = True
    while improved:
        improved = False
        for i in range(D):
            new_routing = routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            new_max = compute_max_load(new_routing)
            if new_max < current_max - 1e-9:
                routing = new_routing
                current_max = new_max
                improved = True
    
    return routing

# EVOLVE-BLOCK-END
