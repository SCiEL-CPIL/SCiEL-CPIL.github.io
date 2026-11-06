"""Rewritten program: ILP via scipy.optimize.milp for densest K2,3-free graph on 20 vertices."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    # Number of vertex pairs
    num_pairs = n * (n - 1) // 2
    # Number of edges
    num_edges = n * (n - 1) // 2
    
    # Build edge index mapping
    edge_list = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = idx
            edge_list.append((u, v))
            idx += 1
    
    # Build pair-to-edges incidence: for each pair of vertices (u,v),
    # which edges are incident to exactly one of u or v (i.e., common neighbors)
    # A common neighbor w of u and v means edges (u,w) and (v,w) both exist.
    # The constraint is: for each pair (u,v), sum over w != u,v of x_{u,w} * x_{v,w} <= 2
    # This is quadratic. We linearize by introducing auxiliary variables.
    #
    # For each pair (u,v) and each potential common neighbor w, let y_{u,v,w} = x_{u,w} * x_{v,w}
    # Then constraint: sum_w y_{u,v,w} <= 2
    #
    # Linearization: y <= x_{u,w}, y <= x_{v,w}, y >= x_{u,w} + x_{v,w} - 1, y >= 0
    #
    # Variables: x_e for each edge e (binary), y_{u,v,w} for each (pair, common neighbor)
    #
    # Total variables: num_edges + sum over pairs of (n - 2)
    
    # Precompute for each pair (u,v), the list of (w, edge_idx_uw, edge_idx_vw)
    pair_data = []  # list of (u, v, [(w, idx_uw, idx_vw), ...])
    for u in range(n):
        for v in range(u + 1, n):
            common_neighbors = []
            for w in range(n):
                if w == u or w == v:
                    continue
                # Edge (u,w) and (v,w)
                if u < w:
                    idx_uw = edge_index[(u, w)]
                else:
                    idx_uw = edge_index[(w, u)]
                if v < w:
                    idx_vw = edge_index[(v, w)]
                else:
                    idx_vw = edge_index[(w, v)]
                common_neighbors.append((w, idx_uw, idx_vw))
            pair_data.append((u, v, common_neighbors))
    
    num_pairs = len(pair_data)
    # Each pair has (n-2) potential common neighbors
    num_y = num_pairs * (n - 2)
    
    total_vars = num_edges + num_y
    
    # Objective: maximize sum of x_e => minimize -sum of x_e
    c = np.zeros(total_vars)
    for e in range(num_edges):
        c[e] = -1.0
    
    # Constraints:
    # 1. For each y_{p,w}: y <= x_uw, y <= x_vw, y >= x_uw + x_vw - 1, y >= 0
    # 2. For each pair p: sum_w y_{p,w} <= 2
    
    # We'll build the constraint matrix A_ub and b_ub for scipy.optimize.milp
    # milp(c, constraints=[LinearConstraint(A, lb, ub)], integrality=..., bounds=...)
    
    # Let's count constraints:
    # For each y variable: 3 constraints (y <= x_uw, y <= x_vw, y >= x_uw + x_vw - 1)
    # Plus y >= 0 is handled by bounds
    # For each pair: 1 constraint (sum y <= 2)
    
    num_y_constraints = num_y * 3
    num_pair_constraints = num_pairs
    total_constraints = num_y_constraints + num_pair_constraints
    
    A = np.zeros((total_constraints, total_vars))
    b_lb = np.zeros(total_constraints)
    b_ub = np.zeros(total_constraints)
    
    # Fill constraints
    row = 0
    for p_idx, (u, v, common_neighbors) in enumerate(pair_data):
        for w_idx, (w, idx_uw, idx_vw) in enumerate(common_neighbors):
            y_var = num_edges + p_idx * (n - 2) + w_idx
            
            # y <= x_uw  =>  y - x_uw <= 0
            A[row, y_var] = 1.0
            A[row, idx_uw] = -1.0
            b_lb[row] = -np.inf
            b_ub[row] = 0.0
            row += 1
            
            # y <= x_vw  =>  y - x_vw <= 0
            A[row, y_var] = 1.0
            A[row, idx_vw] = -1.0
            b_lb[row] = -np.inf
            b_ub[row] = 0.0
            row += 1
            
            # y >= x_uw + x_vw - 1  =>  -y + x_uw + x_vw <= 1
            A[row, y_var] = -1.0
            A[row, idx_uw] = 1.0
            A[row, idx_vw] = 1.0
            b_lb[row] = -np.inf
            b_ub[row] = 1.0
            row += 1
    
    # Pair constraints: sum_w y_{p,w} <= 2
    for p_idx in range(num_pairs):
        for w_idx in range(n - 2):
            y_var = num_edges + p_idx * (n - 2) + w_idx
            A[row, y_var] = 1.0
        b_lb[row] = -np.inf
        b_ub[row] = 2.0
        row += 1
    
    # Integrality: x variables are binary, y variables are also binary (since x are binary, y = x_uw * x_vw is binary)
    integrality = np.ones(total_vars)
    
    # Bounds: all variables in [0, 1]
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    bounds = Bounds(lb, ub)
    
    constraints = [LinearConstraint(A, b_lb, b_ub)]
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            # Extract edges
            edges = []
            for e in range(num_edges):
                if res.x[e] > 0.5:
                    u, v = edge_list[e]
                    edges.append([u, v])
            return edges
        else:
            # Fallback: return the seed solution
            edges = []
            for k in range(5):
                edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
            return edges
    except Exception:
        # Fallback: return the seed solution
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
