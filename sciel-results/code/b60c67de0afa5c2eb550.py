# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

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
    start_time = time.time()
    deadline = 600.0  # 10 minutes, though this should take milliseconds
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    n_edges = n_nodes
    
    # Precompute the clockwise usage matrix A
    # A[i, e] = 1 if demand i uses edge e when routed clockwise
    A = np.zeros((n_demands, n_edges), dtype=float)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path from s to t uses edges s, s+1, ..., t-1
        # If s == t, it's a loop? The problem implies distinct nodes or standard paths.
        # Let's assume standard path. If s == t, load is 0? Or full ring?
        # Usually s != t in these problems. Let's check data.
        # (0,5), (1,7), etc. All s != t.
        
        # Generate edges in CW path
        current = s
        while current != t:
            A[i, current] = 1.0
            current = (current + 1) % n_nodes
            
    # ILP Formulation
    # Variables: x[0..14] (binary), T (continuous)
    # Minimize T
    # Subject to: Load_e <= T for all e
    
    # Load_e = sum_i w_i * uses_{i,e}(x_i)
    # uses_{i,e}(0) = A[i,e]
    # uses_{i,e}(1) = 1 - A[i,e]
    # Load_e = sum_i w_i * (A[i,e]*(1-x_i) + (1-A[i,e])*x_i)
    #         = sum_i w_i * (A[i,e] - x_i*A[i,e] + x_i - x_i*A[i,e])
    #         = sum_i w_i * (A[i,e] + x_i*(1 - 2*A[i,e]))
    
    # Constraint: sum_i w_i * (A[i,e] + x_i*(1 - 2*A[i,e])) <= T
    # => sum_i w_i * (1 - 2*A[i,e]) * x_i - T <= - sum_i w_i * A[i,e]
    
    # Let's use the form: sum_i coeff_i * x_i + 1 * T >= const_e
    # coeff_i = w_i * (1 - 2*A[i,e])
    # const_e = sum_i w_i * A[i,e]
    
    c = np.zeros(n_demands + 1)
    c[-1] = 1.0  # Minimize T
    
    # Constraints: A_ub y <= b_ub  and  A_lb y >= b_lb
    # We have n_edges constraints of type >=
    
    A_lb = np.zeros((n_edges, n_demands + 1))
    b_lb = np.zeros(n_edges)
    
    for e in range(n_edges):
        for i in range(n_demands):
            w = demands[i][2]
            # Coefficient for x_i
            A_lb[e, i] = w * (1.0 - 2.0 * A[i, e])
        # Coefficient for T
        A_lb[e, -1] = 1.0
        # RHS
        b_lb[e] = np.sum([demands[i][2] * A[i, e] for i in range(n_demands)])
    
    # Bounds
    # x_i in {0, 1}
    # T >= 0
    integrality = np.zeros(n_demands + 1)
    integrality[:n_demands] = 1  # x_i are integers
    
    lower_bound = np.zeros(n_demands + 1)
    upper_bound = np.ones(n_demands + 1)
    upper_bound[-1] = np.inf
    
    bounds = Bounds(lb=lower_bound, ub=upper_bound)
    
    # Constraint object
    # milp expects A_ub, b_ub, A_lb, b_lb
    # We only have lower bound constraints (>=)
    constraint = LinearConstraint(A_lb, lb=b_lb, ub=np.inf)
    
    try:
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:
            x_solution = res.x[:n_demands]
            # Ensure clean 0/1
            x_solution = (x_solution > 0.5).astype(float)
            return x_solution
        else:
            # Fallback: all clockwise (0)
            return np.zeros(n_demands)
            
    except Exception:
        # Fallback: all clockwise (0)
        return np.zeros(n_demands)

# EVOLVE-BLOCK-END
