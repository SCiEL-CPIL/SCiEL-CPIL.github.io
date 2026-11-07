"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP using a 
    min-cost flow / LP relaxation approach to encode optimal routing structure.
    
    Args:
        distance_matrix: (n, n) symmetric distance matrix
        coordinates: (n, 2) coordinates
        demands: (n,) demands (node 0 is depot with demand 0)
        capacity: vehicle capacity
        
    Returns:
        (n, n) heuristic matrix where larger = more attractive
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    d = distance_matrix.copy().astype(np.float64)
    np.fill_diagonal(d, np.inf)
    d = np.where(np.isfinite(d), d, 1e10)
    
    # Node 0 is depot, nodes 1..n-1 are customers
    num_customers = n - 1
    
    # Build edge list for all (i, j) pairs where i != j
    # We'll model a relaxed flow problem: each customer must be served,
    # and vehicles depart from depot, visit customers, return to depot.
    # The LP relaxation allows fractional flow, giving us a "soft" optimal structure.
    
    # Number of edges in complete directed graph (excluding self-loops)
    edge_count = n * (n - 1)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j))
    
    # Map edge (i,j) to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Objective: minimize total distance cost
    c = np.array([d[i, j] for i, j in edges], dtype=np.float64)
    
    # Variables: x[e] >= 0 for each edge e
    
    # Constraints:
    # 1. Flow conservation: for each customer node k (1..n-1), 
    #    sum_in(x) - sum_out(x) = 0 (customer must have equal in/out flow)
    # 2. For depot (node 0): sum_out(x) - sum_in(x) = num_vehicles (or just balance)
    #    Actually, for a proper flow: sum_out(depot) - sum_in(depot) = number of routes
    #    But we don't know number of routes. Instead, use:
    #    Each customer has exactly 1 unit of demand to be served.
    #    Flow represents "visits" - each customer should have flow through it.
    
    # Simpler formulation: 
    # - Each customer j must have at least 1 unit of flow passing through 
    #   (i.e., sum of incoming flow to j >= 1, and sum of outgoing flow from j >= 1)
    # - Depot can have any net outflow
    # - Minimize total cost
    
    # Let's use: for each customer j, sum_{i != j} x[i,j] >= 1 (at least one incoming)
    # and sum_{j != i} x[i,j] >= 1 (at least one outgoing)
    # This ensures each customer is visited at least once.
    
    # Build A_ub and b_ub for: A_ub @ x <= b_ub
    # And A_eq and b_eq for: A_eq @ x = b_eq
    
    # Inequality constraints: for each customer j (1..n-1):
    #   sum_{i != j} x[i,j] >= 1  =>  -sum_{i != j} x[i,j] <= -1
    #   sum_{j != i} x[i,j] >= 1  =>  -sum_{j != i} x[i,j] <= -1
    
    num_constraints = 2 * num_customers
    A_ub = np.zeros((num_constraints, edge_count), dtype=np.float64)
    b_ub = np.zeros(num_constraints, dtype=np.float64)
    
    for j in range(1, n):
        # Incoming constraint: sum_{i != j} x[i,j] >= 1
        row_in = 2 * (j - 1)
        for i in range(n):
            if i != j:
                if (i, j) in edge_idx:
                    A_ub[row_in, edge_idx[(i, j)]] = -1.0
        b_ub[row_in] = -1.0
        
        # Outgoing constraint: sum_{j' != i} x[i,j'] >= 1
        row_out = 2 * (j - 1) + 1
        for j_prime in range(n):
            if j_prime != j:
                if (j, j_prime) in edge_idx:
                    A_ub[row_out, edge_idx[(j, j_prime)]] = -1.0
        b_ub[row_out] = -1.0
    
    # Bounds: x >= 0, no upper bound (or large upper bound)
    bounds = [(0, None)] * edge_count
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_opt = result.x
        else:
            # Fallback to inverse distance heuristic
            x_opt = None
    except Exception:
        x_opt = None
    
    # Build heuristic matrix
    H = np.zeros((n, n))
    
    if x_opt is not None and time.time() < deadline:
        # Use the optimal flow to guide the heuristic
        # Higher flow on an edge means it's more "important" in optimal solutions
        # We want higher heuristic for edges with higher flow, but also penalize long edges
        
        # Smooth the flow to avoid zero values
        x_smooth = x_opt + 1e-6
        
        # Heuristic: combine flow importance and inverse distance
        # H[i,j] = (flow[i,j]) / (distance[i,j])^beta
        # This encourages edges that are both important in optimal flow AND short
        
        beta = 0.8
        for idx, (i, j) in enumerate(edges):
            if d[i, j] < 1e-10:
                H[i, j] = 1e-9
            else:
                H[i, j] = x_smooth[idx] / (d[i, j] ** beta)
        
        # Add a small base component to ensure all edges have some attractiveness
        base = 1.0 / (d ** 1.2)
        np.fill_diagonal(base, 0)
        H = 0.7 * H + 0.3 * base
        
    else:
        # Fallback: pure inverse distance heuristic
        alpha = 1.5
        H = 1.0 / (d ** alpha)
        np.fill_diagonal(H, 1e-9)
    
    # Ensure diagonal is small
    np.fill_diagonal(H, 1e-9)
    
    # Normalize
    H_max = np.max(H)
    if H_max > 0 and np.isfinite(H_max):
        H = H / H_max
    
    # Ensure all entries are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
