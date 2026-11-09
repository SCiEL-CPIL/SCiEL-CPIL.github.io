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
    deadline = start_time + 580  # safety margin before 630s limit

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
        # ccw_weights[i, e] = w if e not in CW, else 0.
        ccw_weights[i, :] = w - cw_weights[i, :]

    # base_load[e] = sum of cw_weights[i, e] for all i
    # This is the load if all demands are routed CW.
    base_load = np.sum(cw_weights, axis=0)
    
    # delta_load[i, e] = ccw_weights[i, e] - cw_weights[i, e]
    # If demand i is routed CCW (bit 1), we add delta_load[i, e] to the load.
    # If demand i is routed CW (bit 0), we add 0.
    # Load[e] = base_load[e] + sum_{i: bit_i=1} delta_load[i, e]
    delta_load = ccw_weights - cw_weights

    # --- Strategy 1: Linear Programming Relaxation ---
    # Variables: x_0 ... x_{D-1} (directions), L (max load)
    # Minimize L
    # Subject to:
    #   For each edge e: base_load[e] + sum_i delta_load[i, e] * x_i <= L
    #   0 <= x_i <= 1
    #   L is free (or >= 0)
    
    # c vector: [x_0, ..., x_{D-1}, L]
    c = np.zeros(D + 1)
    c[-1] = 1.0  # Minimize L
    
    # A_ub * vars <= b_ub
    # For each edge e: sum_i delta_load[i, e] * x_i - L <= -base_load[e]
    # A_ub has shape (n_nodes, D + 1)
    A_ub = np.zeros((n_nodes, D + 1))
    b_ub = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        A_ub[e, :D] = delta_load[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -base_load[e]
    
    # Bounds: 0 <= x_i <= 1, L >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x_sol = res.x[:D]
            # Round to nearest binary
            binary_sol = np.where(x_sol >= 0.5, 1.0, 0.0)
            
            # Verify the solution
            load = base_load + binary_sol @ delta_load
            max_load = np.max(load)
            
            # Store best so far
            best_max_load = max_load
            best_routing = binary_sol
        else:
            # Fallback to heuristic if LP fails
            best_routing = np.zeros(D)
            load = base_load
            best_max_load = np.max(load)
    except Exception:
        best_routing = np.zeros(D)
        load = base_load
        best_max_load = np.max(load)

    # --- Strategy 2: Local Search Polish ---
    # Try to improve the solution found by LP or heuristic using local search.
    # This is important because LP relaxation might give a non-integer solution
    # that rounds poorly, or the LP might not find the global optimum due to
    # numerical issues or specific problem structure.
    
    def compute_max_load(routing):
        """Compute the maximum edge load given a routing assignment (binary array)."""
        # routing is a 0/1 array
        load = base_load + routing @ delta_load
        return np.max(load)

    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Try flipping single demands
        for i in range(D):
            if time.time() > deadline:
                break
            old_val = best_routing[i]
            best_routing[i] = 1.0 - old_val
            new_max_load = compute_max_load(best_routing)
            if new_max_load < best_max_load - 1e-9:
                best_max_load = new_max_load
                improved = True
            else:
                best_routing[i] = old_val
        
        # Try flipping pairs of demands
        if time.time() > deadline:
            break
        for i in range(D):
            if time.time() > deadline:
                break
            for j in range(i + 1, D):
                if time.time() > deadline:
                    break
                old_i = best_routing[i]
                old_j = best_routing[j]
                best_routing[i] = 1.0 - old_i
                best_routing[j] = 1.0 - old_j
                new_max_load = compute_max_load(best_routing)
                if new_max_load < best_max_load - 1e-9:
                    best_max_load = new_max_load
                    improved = True
                else:
                    best_routing[i] = old_i
                    best_routing[j] = old_j

    # --- Strategy 3: Random Restarts (if time permits) ---
    # If we still have time, try a few random restarts to escape local optima.
    if time.time() < deadline - 5:
        np.random.seed(42)
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            # Random initialization
            routing = np.random.choice([0.0, 1.0], size=D)
            current_max_load = compute_max_load(routing)
            
            # Local search from this random start
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                progress = False
                # Try single flips
                for i in range(D):
                    if time.time() > deadline - 5:
                        break
                    old_val = routing[i]
                    routing[i] = 1.0 - old_val
                    new_max_load = compute_max_load(routing)
                    if new_max_load < current_max_load - 1e-9:
                        current_max_load = new_max_load
                        progress = True
                    else:
                        routing[i] = old_val
                
                # Try pair flips
                if time.time() > deadline - 5:
                    break
                for i in range(D):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, D):
                        if time.time() > deadline - 5:
                            break
                        old_i = routing[i]
                        old_j = routing[j]
                        routing[i] = 1.0 - old_i
                        routing[j] = 1.0 - old_j
                        new_max_load = compute_max_load(routing)
                        if new_max_load < current_max_load - 1e-9:
                            current_max_load = new_max_load
                            progress = True
                        else:
                            routing[i] = old_i
                            routing[j] = old_j
                
                if not progress:
                    break
            
            if current_max_load < best_max_load - 1e-9:
                best_max_load = current_max_load
                best_routing = routing.copy()

    return best_routing

# EVOLVE-BLOCK-END
