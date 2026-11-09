# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import math
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

    # Precompute edge masks as numpy arrays for fast vectorized load computation
    # For demand d with source s, target t:
    # Clockwise: edges s, s+1, ..., t-1 (mod 10)
    # Counter-clockwise: all other edges
    cw_mask = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_mask = np.zeros((D, n_nodes), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            cw_mask[i, :] = 1.0
        else:
            node = s
            while node != t:
                cw_mask[i, node] = 1.0
                node = (node + 1) % n_nodes
        ccw_mask[i] = 1.0 - cw_mask[i]

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    def compute_loads(routing):
        """Compute edge loads given a binary routing array (0=CW, 1=CCW)."""
        # routing: shape (D,), values 0 or 1
        # load on edge e = sum over demands i of w_i * mask[i,e]
        # where mask = cw_mask[i] if routing[i]==0 else ccw_mask[i]
        # = sum_i w_i * (cw_mask[i] * (1-routing[i]) + ccw_mask[i] * routing[i])
        # = sum_i w_i * (cw_mask[i] + (ccw_mask[i] - cw_mask[i]) * routing[i])
        diff = ccw_mask - cw_mask  # shape (D, n_nodes)
        base = (weights[:, None] * cw_mask).sum(axis=0)  # shape (n_nodes,)
        loads = base + (weights[:, None] * diff * routing[:, None]).sum(axis=0)
        return loads

    def compute_max_load(routing):
        return np.max(compute_loads(routing))

    def local_search(routing, max_iters=500):
        """
        Local search with 1-way, 2-way, and 3-way flips.
        Returns (routing, max_load) at convergence.
        """
        current_loads = compute_loads(routing)
        current_max = np.max(current_loads)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            progress = False
            
            # 1-way flips
            for i in range(D):
                if time.time() > deadline:
                    break
                old = routing[i]
                routing[i] = 1.0 - old
                new_loads = compute_loads(routing)
                new_max = np.max(new_loads)
                if new_max < current_max - 1e-9:
                    current_max = new_max
                    current_loads = new_loads
                    progress = True
                else:
                    routing[i] = old
            
            # 2-way flips
            if not progress:
                for i in range(D):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, D):
                        if time.time() > deadline:
                            break
                        oi, oj = routing[i], routing[j]
                        routing[i], routing[j] = 1.0 - oi, 1.0 - oj
                        new_loads = compute_loads(routing)
                        new_max = np.max(new_loads)
                        if new_max < current_max - 1e-9:
                            current_max = new_max
                            current_loads = new_loads
                            progress = True
                        else:
                            routing[i], routing[j] = oi, oj
                        if progress:
                            break
                    if progress:
                        break
            
            # 3-way flips
            if not progress:
                for i in range(D):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, D):
                        if time.time() > deadline:
                            break
                        for k in range(j + 1, D):
                            if time.time() > deadline:
                                break
                            oi, oj, ok = routing[i], routing[j], routing[k]
                            routing[i], routing[j], routing[k] = 1.0-oi, 1.0-oj, 1.0-ok
                            new_loads = compute_loads(routing)
                            new_max = np.max(new_loads)
                            if new_max < current_max - 1e-9:
                                current_max = new_max
                                current_loads = new_loads
                                progress = True
                            else:
                                routing[i], routing[j], routing[k] = oi, oj, ok
                            if progress:
                                break
                        if progress:
                            break
                    if progress:
                        break
            
            if not progress:
                break
        
        return routing, current_max

    def simulated_annealing(routing, max_iters=2000):
        """
        Simulated annealing to escape local optima.
        Returns (routing, max_load).
        """
        current_loads = compute_loads(routing)
        current_max = np.max(current_loads)
        T0 = 2.0
        T_final = 0.001
        cooling = (T_final / T0) ** (1.0 / max_iters)
        T = T0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            T *= cooling
            
            # Random 1 or 2 flip
            n_flips = 1 if np.random.random() < 0.6 else 2
            if n_flips == 1:
                i = np.random.randint(D)
                old = routing[i]
                routing[i] = 1.0 - old
                new_loads = compute_loads(routing)
                new_max = np.max(new_loads)
                delta = new_max - current_max
                if delta < -1e-9 or (T > 1e-6 and np.random.random() < math.exp(-delta / T)):
                    current_max = new_max
                    current_loads = new_loads
                else:
                    routing[i] = old
            else:
                idx = np.random.choice(D, size=2, replace=False)
                oi, oj = routing[idx[0]], routing[idx[1]]
                routing[idx[0]], routing[idx[1]] = 1.0-oi, 1.0-oj
                new_loads = compute_loads(routing)
                new_max = np.max(new_loads)
                delta = new_max - current_max
                if delta < -1e-9 or (T > 1e-6 and np.random.random() < math.exp(-delta / T)):
                    current_max = new_max
                    current_loads = new_loads
                else:
                    routing[idx[0]], routing[idx[1]] = oi, oj
        
        return routing, current_max

    # Phase 1: Solve the min-max LP exactly using HiGHS
    # Variables: x_0, ..., x_{D-1} (binary routing), L (max load)
    # Minimize L
    # Subject to: for each edge e, sum_i w_i * mask_i_e <= L
    # where mask_i_e = cw_mask[i,e] if x_i=0, ccw_mask[i,e] if x_i=1
    # = cw_mask[i,e] + (ccw_mask[i,e] - cw_mask[i,e]) * x_i
    # So constraint for edge e: sum_i w_i * (cw_mask[i,e] + diff[i,e]*x_i) <= L
    # => -L + sum_i w_i * diff[i,e] * x_i <= -sum_i w_i * cw_mask[i,e]
    # => -L + sum_i w_i * diff[i,e] * x_i + sum_i w_i * cw_mask[i,e] <= 0
    # Actually let me rewrite:
    # load_e = base_e + sum_i w_i * diff[i,e] * x_i <= L
    # => -L + sum_i w_i * diff[i,e] * x_i <= -base_e
    # where base_e = sum_i w_i * cw_mask[i,e]
    
    # Decision variables: [x_0, ..., x_{D-1}, L]
    # Objective: minimize L => c = [0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[-1] = 1.0  # minimize L
    
    # Inequality constraints: A_ub @ vars <= b_ub
    # For each edge e: -L + sum_i w_i * diff[i,e] * x_i <= -base_e
    base = (weights[:, None] * cw_mask).sum(axis=0)  # shape (n_nodes,)
    diff = ccw_mask - cw_mask  # shape (D, n_nodes)
    
    # A_ub has n_nodes rows, D+1 columns
    A_ub = np.zeros((n_nodes, D + 1))
    b_ub = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        # -L + sum_i w_i * diff[i,e] * x_i <= -base_e
        A_ub[e, :D] = weights * diff[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -base[e]
    
    # Bounds: x_i in [0, 1], L in [0, infinity]
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Solve LP with HiGHS
    lp_result = None
    try:
        lp_result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'time_limit': 300}
        )
    except Exception:
        lp_result = None
    
    # Phase 2: Extract solution from LP or use local search
    best_routing = np.zeros(D)
    best_max_load = compute_max_load(best_routing)
    
    if lp_result is not None and lp_result.success:
        # x_i values from LP solution
        lp_x = lp_result.x[:D]
        lp_L = lp_result.x[D]
        
        # Round to binary
        lp_routing = np.where(lp_x >= 0.5, 1.0, 0.0)
        lp_max_load = compute_max_load(lp_routing)
        
        if lp_max_load < best_max_load - 1e-9:
            best_max_load = lp_max_load
            best_routing = lp_routing.copy()
        
        # Also try the other rounding
        lp_routing2 = np.where(lp_x > 0.5, 1.0, 0.0)
        if not np.array_equal(lp_routing, lp_routing2):
            lp_max_load2 = compute_max_load(lp_routing2)
            if lp_max_load2 < best_max_load - 1e-9:
                best_max_load = lp_max_load2
                best_routing = lp_routing2.copy()
    
    # Try all-CW and all-CCW as baselines
    all_ccw = np.ones(D)
    ml = compute_max_load(all_ccw)
    if ml < best_max_load:
        best_max_load = ml
        best_routing = all_ccw.copy()
    
    # Local search from best solution
    r, ml = local_search(best_routing.copy(), max_iters=200)
    if ml < best_max_load - 1e-9:
        best_max_load = ml
        best_routing = r.copy()

    # Phase 3: Random restarts with local search + simulated annealing
    np.random.seed(42)
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        routing = np.random.choice([0.0, 1.0], size=D)
        
        # Local search
        routing, ml = local_search(routing, max_iters=100)
        
        # Simulated annealing from local optimum
        routing, ml_sa = simulated_annealing(routing, max_iters=500)
        
        # Final local search polish
        routing, ml_final = local_search(routing, max_iters=100)
        
        if ml_final < best_max_load - 1e-9:
            best_max_load = ml_final
            best_routing = routing.copy()

    # Phase 4: Final polish with 3-way flips on the best solution
    best_routing, best_max_load = local_search(best_routing.copy(), max_iters=200)

    return best_routing


# EVOLVE-BLOCK-END
