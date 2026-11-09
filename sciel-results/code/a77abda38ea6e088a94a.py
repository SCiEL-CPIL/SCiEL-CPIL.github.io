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
    cw_matrix = np.zeros((D, n_edges))
    ccw_matrix = np.zeros((D, n_edges))

    for d, (s, t, w) in enumerate(demands):
        if s == t:
            continue
        if s < t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[d, e] = w
        
        for e in range(n_edges):
            if cw_matrix[d, e] == 0:
                ccw_matrix[d, e] = w

    # Formulate as MILP
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    A_ub = np.zeros((n_edges, D + 1))
    b_ub = np.zeros(n_edges)
    
    for e in range(n_edges):
        for d in range(D):
            A_ub[e, d] = -(cw_matrix[d, e] - ccw_matrix[d, e])
        A_ub[e, D] = -1.0
        b_ub[e] = -np.sum(cw_matrix[:, e])
    
    bounds = Bounds(lb=np.concatenate((np.zeros(D), [0])), 
                    ub=np.concatenate((np.ones(D), [np.inf])))
    
    integrality = np.concatenate((np.ones(D, dtype=int), [0]))
    
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    
    routing = None
    try:
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if result.success:
            routing = np.round(result.x[:D])
            routing[routing < 0.5] = 0.0
            routing[routing >= 0.5] = 1.0
    except Exception:
        pass

    if routing is None:
        routing = np.zeros(D)

    # Local Search to improve the solution
    def get_max_load(r):
        loads = np.zeros(n_edges)
        for d in range(D):
            if r[d] < 0.5:
                loads += cw_matrix[d]
            else:
                loads += ccw_matrix[d]
        return np.max(loads)

    current_load = get_max_load(routing)
    improved = True
    while improved:
        improved = False
        for d in range(D):
            # Try flipping demand d
            new_routing = routing.copy()
            new_routing[d] = 1.0 - new_routing[d]
            new_load = get_max_load(new_routing)
            if new_load < current_load:
                routing = new_routing
                current_load = new_load
                improved = True
    
    return routing

# EVOLVE-BLOCK-END
