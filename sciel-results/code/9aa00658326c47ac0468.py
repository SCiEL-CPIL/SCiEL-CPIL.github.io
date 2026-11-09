# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Uses Integer Linear Programming (ILP) for an optimal solution.
    """
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute clockwise edges for each demand
    cw_edges = []
    for s, t, w in demands:
        edges = set()
        e = s
        while e != t:
            edges.add(e)
            e = (e + 1) % N
        cw_edges.append(edges)
    
    # Counter-clockwise uses all other edges
    all_edges = set(range(N))
    ccw_edges = []
    for ce in cw_edges:
        ccw_edges.append(all_edges - ce)
    
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # ILP Formulation:
    # Variables: x[0..D-1] (binary, 1=CCW, 0=CW), T (continuous, max load)
    # Total vars: D + 1 = 16
    # Objective: Minimize T -> c = [0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Constraints:
    # For each edge j (0..N-1), Load_j <= T
    # Load_j = sum_{i in CW_j} w_i * (1 - x_i) + sum_{i in CCW_j} w_i * x_i
    # Load_j = C_j - sum_{i in CW_j} w_i x_i + sum_{i in CCW_j} w_i x_i
    # where C_j = sum_{i in CW_j} w_i
    # Load_j - T <= 0  =>  T - Load_j >= 0
    # T - (C_j + sum_{i} A_{ji} x_i) >= 0
    # T - sum_{i} A_{ji} x_i >= C_j
    # where A_{ji} = -w_i if i in CW_j, w_i if i in CCW_j, else 0
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    for j in range(N):
        row = np.zeros(D + 1)
        # Coefficient for T is 1
        row[D] = 1.0
        
        C_j = 0.0
        for i in range(D):
            if j in cw_edges[i]:
                # Demand i is CW, uses edge j if x_i=0
                # Contribution to Load_j: w_i * (1 - x_i) = w_i - w_i x_i
                # So coeff of x_i in Load_j is -w_i
                # Constraint: T - Load_j >= 0 => T - (w_i - w_i x_i) >= ... 
                # T + w_i x_i - w_i >= ...
                # Wait, let's stick to T - sum(A_ij x_j) >= C_j
                # Load_j = C_j + sum(A_ij x_j)
                # If i in CW_j: Load_j gets w_i when x_i=0. 
                # Load_j = ... + w_i(1-x_i) = ... + w_i - w_i x_i
                # So A_ij = -w_i.
                # And C_j accumulates w_i.
                C_j += weights[i]
                row[i] = -(-weights[i]) # coeff in LHS T - sum(A x) is -A. 
                # Let's re-derive carefully.
                # Inequality: T - Load_j >= 0
                # Load_j = C_j + sum_k A_k x_k
                # T - C_j - sum_k A_k x_k >= 0
                # T - sum_k A_k x_k >= C_j
                # LHS vector: [ -A_0, -A_1, ..., -A_{D-1}, 1 ]
                # RHS lower: C_j, upper: inf
                
                # For i in CW_j: A_i = -w_i. So coeff in LHS is -(-w_i) = w_i.
                row[i] = weights[i]
            elif j in ccw_edges[i]:
                # Demand i is CCW, uses edge j if x_i=1
                # Contribution to Load_j: w_i * x_i
                # A_i = w_i.
                # Coeff in LHS is -w_i.
                row[i] = -weights[i]
            else:
                # Edge j not used by demand i in either direction? 
                # In a ring, every demand uses either CW path or CCW path.
                # CW path and CCW path are disjoint (except if s=t, but here s!=t).
                # Actually, CW path edges + CCW path edges = all edges.
                # So if not in CW, it is in CCW.
                pass
        
        A_rows.append(row)
        b_lower.append(C_j)
        b_upper.append(np.inf)
    
    A = np.array(A_rows)
    
    # Integrality: first D are binary, last is continuous
    integrality = np.ones(D + 1)
    integrality[D] = 0
    
    # Bounds
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    
    res = milp(c=c, constraints=LinearConstraint(A, lb=np.array(b_lower), ub=np.array(b_upper)), 
               integrality=integrality, bounds=Bounds(lb, ub))
    
    if res.status == 0:
        x_opt = res.x[:D]
        # Round to 0 or 1 to be safe
        x_opt = np.where(x_opt > 0.5, 1.0, 0.0)
        return x_opt
    else:
        # Fallback to local search if ILP fails (should not happen)
        return _fallback_local_search(N, D, demands)

def _fallback_local_search(N, D, demands):
    # ... (Keep original logic as fallback if needed, but ILP should work)
    cw_edges = []
    for s, t, w in demands:
        edges = set()
        e = s
        while e != t:
            edges.add(e)
            e = (e + 1) % N
        cw_edges.append(edges)
    all_edges = set(range(N))
    ccw_edges = []
    for ce in cw_edges:
        ccw_edges.append(all_edges - ce)
    weights = np.array([d[2] for d in demands], dtype=float)
    
    def compute_loads(routing):
        loads = np.zeros(N)
        for i in range(D):
            if routing[i] < 0.5:
                for e in cw_edges[i]:
                    loads[e] += weights[i]
            else:
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
        return loads
    
    def max_load(routing):
        return np.max(compute_loads(routing))
        
    best_routing = np.zeros(D)
    best_ml = max_load(best_routing)
    improved = True
    while improved:
        improved = False
        for i in range(D):
            new_routing = best_routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            ml = max_load(new_routing)
            if ml < best_ml - 1e-12:
                best_routing = new_routing
                best_ml = ml
                improved = True
    return best_routing

# EVOLVE-BLOCK-END
