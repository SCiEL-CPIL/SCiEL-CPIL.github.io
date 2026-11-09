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
    
    def compute_loads(routing):
        """Compute edge loads given a routing vector (0=cw, 1=ccw)."""
        loads = np.zeros(N)
        for i in range(D):
            if routing[i] < 0.5:  # clockwise
                for e in cw_edges[i]:
                    loads[e] += weights[i]
            else:  # counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
        return loads
    
    def max_load(routing):
        return np.max(compute_loads(routing))
    
    # ILP Formulation:
    # Variables: x[0..D-1] (binary, 1=CCW, 0=CW), T (continuous, max load)
    # Total vars: D + 1 = 16
    # Objective: minimize T
    # Constraints: For each edge e, sum over demands using e of weight_i * (direction) <= T
    # If demand i goes CW (x_i=0), it uses cw_edges[i]
    # If demand i goes CCW (x_i=1), it uses ccw_edges[i]
    # Load on edge e = sum_i weight_i * (1 if i uses e)
    # For CW: uses e if e in cw_edges[i], coefficient for x_i is 0
    # For CCW: uses e if e in ccw_edges[i], coefficient for x_i is weight_i
    # So load_e = sum_i weight_i * [e in cw_edges[i] * (1-x_i) + e in ccw_edges[i] * x_i]
    #           = sum_i weight_i * [e in cw_edges[i]] + sum_i weight_i * [e in ccw_edges[i]] * x_i - sum_i weight_i * [e in cw_edges[i]] * x_i
    #           = base_e + sum_i weight_i * ([e in ccw_edges[i]] - [e in cw_edges[i]]) * x_i
    # Constraint: base_e + sum_i c_{e,i} * x_i <= T for all e
    # i.e., T - sum_i c_{e,i} * x_i >= base_e
    
    # Build constraint matrix A_ub such that A_ub @ vars <= b_ub
    # vars = [x_0, ..., x_{D-1}, T]
    # For each edge e: base_e + sum_i c_{e,i} x_i - T <= 0
    # => -T + sum_i c_{e,i} x_i <= -base_e
    
    A = np.zeros((N, D + 1))
    b = np.zeros(N)
    
    for e in range(N):
        base_e = 0.0
        for i in range(D):
            if e in cw_edges[i]:
                base_e += weights[i]
            # c_{e,i} = weight_i if e in ccw_edges[i] else 0 - (weight_i if e in cw_edges[i] else 0)
            # Since e is either in cw or ccw (not both, since they partition all_edges):
            # if e in cw_edges[i]: c = 0 - weight_i = -weight_i
            # if e in ccw_edges[i]: c = weight_i - 0 = weight_i
            if e in ccw_edges[i]:
                A[e, i] = weights[i]
            elif e in cw_edges[i]:
                A[e, i] = -weights[i]
        A[e, D] = -1.0  # coefficient for T
        b[e] = -base_e
    
    # Bounds: x_i in [0,1], T in [0, inf)
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf  # T unbounded above
    
    # Objective: minimize T
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Integrality: x_i binary, T continuous
    integrality = np.zeros(D + 1)
    integrality[:D] = 1  # binary
    integrality[D] = 0   # continuous
    
    # Solve ILP
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    try:
        result = milp(
            c=c,
            constraints=LinearConstraint(A, -np.inf, b),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if result.success and result.x is not None:
            x_opt = result.x[:D]
            # Convert to routing vector: x_i = 1 means CCW, x_i = 0 means CW
            best_routing = x_opt.copy()
            # Ensure values are clearly 0 or 1
            best_routing = np.where(best_routing > 0.5, 1.0, 0.0)
            return best_routing
        else:
            # Fallback to local search
            pass
    except Exception:
        pass
    
    # Fallback: local search with random restarts
    def _fallback_local_search():
        deadline_fb = time.time() + 580
        
        # Start with all clockwise
        best_routing = np.zeros(D)
        best_ml = max_load(best_routing)
        
        # Local search: flip one demand at a time
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
            if time.time() > deadline_fb - 30:
                break
        
        # Additional passes with random restarts if time permits
        rng = np.random.default_rng(42)
        for restart in range(100):
            if time.time() > deadline_fb - 30:
                break
            routing = rng.random(D)
            ml = max_load(routing)
            
            improved = True
            while improved:
                improved = False
                for i in range(D):
                    new_routing = routing.copy()
                    new_routing[i] = 1.0 - new_routing[i]
                    ml_new = max_load(new_routing)
                    if ml_new < ml - 1e-12:
                        routing = new_routing
                        ml = ml_new
                        improved = True
                if time.time() > deadline_fb - 30:
                    break
            
            if ml < best_ml - 1e-12:
                best_routing = routing
                best_ml = ml
        
        return best_routing
    
    return _fallback_local_search()


# EVOLVE-BLOCK-END
