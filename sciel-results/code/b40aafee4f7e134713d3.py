# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

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
    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Build the edge-usage matrix for clockwise routing
    # A[demand_idx, edge_idx] = weight if clockwise route of demand uses edge, else 0
    # For counter-clockwise, we use the complement edges
    cw_matrix = np.zeros((D, n_edges))
    ccw_matrix = np.zeros((D, n_edges))

    for d, (s, t, w) in enumerate(demands):
        if s == t:
            # No edges used in either direction (shouldn't happen here)
            continue
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        if s < t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[d, e] = w
        
        # Counter-clockwise uses all other edges
        for e in range(n_edges):
            if cw_matrix[d, e] == 0:
                ccw_matrix[d, e] = w

    # Formulate as MILP:
    # Variables: x[0..14] (routing decisions, 0 or 1), x[15] = max load L
    # Minimize: L
    # Subject to:
    #   For each edge e: sum_d (cw_matrix[d,e] * (1-x_d) + ccw_matrix[d,e] * x_d) <= L
    #   0 <= x_d <= 1
    #
    # Expanding: sum_d cw_matrix[d,e] - sum_d cw_matrix[d,e]*x_d + sum_d ccw_matrix[d,e]*x_d <= L
    #            sum_d (ccw_matrix[d,e] - cw_matrix[d,e]) * x_d - L <= -sum_d cw_matrix[d,e]
    
    # In standard form for milp: minimize c^T x subject to constraints
    # Let variables be: x_0,...,x_14 (routing), x_15 = L
    # Objective: minimize L -> c = [0,...,0, 1]
    
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize L
    
    # Constraints for each edge:
    # sum_d (cw_matrix[d,e] - ccw_matrix[d,e]) * x_d + L >= sum_d cw_matrix[d,e]
    # => -sum_d (cw_matrix[d,e] - ccw_matrix[d,e]) * x_d - L <= -sum_d cw_matrix[d,e]
    # Let A_ub be the matrix for the <= constraints.
    
    A_ub = np.zeros((n_edges, D + 1))
    b_ub = np.zeros(n_edges)
    
    for e in range(n_edges):
        for d in range(D):
            A_ub[e, d] = -(cw_matrix[d, e] - ccw_matrix[d, e])
        A_ub[e, D] = -1.0
        b_ub[e] = -np.sum(cw_matrix[:, e])
    
    # Bounds: 0 <= x_d <= 1 for d in 0..14, L >= 0
    bounds = Bounds(lb=np.concatenate((np.zeros(D), [0])), 
                    ub=np.concatenate((np.ones(D), [np.inf])))
    
    # Integrality: x_0...x_14 are binary (0 or 1), x_15 is continuous
    integrality = np.concatenate((np.ones(D, dtype=int), [0]))
    
    # Define linear constraints
    # A_ub x <= b_ub
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    
    # Solve MILP
    try:
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if not result.success:
             # Fallback: all clockwise
             return np.zeros(D)
        
        routing = result.x[:D]
        
        # Ensure values are 0 or 1
        # milp should return integers for integer variables
        routing = np.round(routing)
        routing[routing < 0.5] = 0.0
        routing[routing >= 0.5] = 1.0
        
        return routing
    except Exception:
        # Fallback if milp fails or is not available/behaves unexpectedly
        # Use the LP relaxation rounded as a heuristic
        from scipy.optimize import linprog
        bounds_lp = [(0, 1)] * D + [(0, None)]
        result_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if not result_lp.success:
            return np.zeros(D)
        routing = result_lp.x[:D]
        routing = np.round(routing)
        routing[routing < 0.5] = 0.0
        routing[routing >= 0.5] = 1.0
        return routing

# EVOLVE-BLOCK-END
