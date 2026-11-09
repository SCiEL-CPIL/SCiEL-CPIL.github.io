# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

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
    DEADLINE = 590  # safety margin before 600s kill

    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute CW and CCW edge incidence for each demand
    # CW from s to t: edges s, s+1, ..., t-1 (mod n)
    # CCW: all other edges
    cw_mask = np.zeros((D, n_edges), dtype=np.float64)
    ccw_mask = np.zeros((D, n_edges), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)

    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # CW edges
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        for e in cw_edges:
            cw_mask[i, e] = 1.0
        # CCW is complement
        ccw_mask[i] = 1.0 - cw_mask[i]

    # Precompute weight * mask for fast addition
    cw_w = cw_mask * weights[:, np.newaxis]
    ccw_w = ccw_mask * weights[:, np.newaxis]

    def compute_load(routing):
        """Compute edge loads given routing vector (0=CW, 1=CCW)."""
        # routing: shape (D,) with 0 or 1
        # load = sum over demands of weight * mask
        # If routing[i]==0, use cw_w[i]; if 1, use ccw_w[i]
        load = np.zeros(n_edges, dtype=np.float64)
        for i in range(D):
            if routing[i] < 0.5:
                load += cw_w[i]
            else:
                load += ccw_w[i]
        return load

    def max_load(routing):
        return np.max(compute_load(routing))

    # --- Phase 1: Greedy initialization + local search ---
    def local_search(initial_routing, max_iters=50000):
        """Hill-climbing with first-improvement, then best-improvement polish."""
        best_routing = initial_routing.copy()
        best_ml = max_load(best_routing)
        best_load = compute_load(best_routing)
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try flipping each demand
            for i in range(D):
                # Try flipping demand i
                new_routing = best_routing.copy()
                new_routing[i] = 1.0 - new_routing[i]
                
                # Compute new load efficiently
                if best_routing[i] < 0.5:
                    # Was CW, now CCW: subtract cw_w[i], add ccw_w[i]
                    new_load = best_load - cw_w[i] + ccw_w[i]
                else:
                    # Was CCW, now CW: subtract ccw_w[i], add cw_w[i]
                    new_load = best_load - ccw_w[i] + cw_w[i]
                
                new_ml = np.max(new_load)
                if new_ml < best_ml - 1e-12:
                    best_routing = new_routing
                    best_ml = new_ml
                    best_load = new_load
                    improved = True
                    break  # Restart scan after improvement (first-improvement)
            
            if not improved:
                break
        
        return best_routing, best_ml

    # Try multiple random initializations
    best_routing_overall = np.zeros(D, dtype=np.float64)
    best_ml_overall = max_load(best_routing_overall)
    
    # Start with all CW
    r0 = np.zeros(D, dtype=np.float64)
    r0, ml0 = local_search(r0)
    if ml0 < best_ml_overall:
        best_ml_overall = ml0
        best_routing_overall = r0.copy()
    
    # Random restarts
    np.random.seed(42)
    n_restarts = 200
    for restart in range(n_restarts):
        if time.time() > DEADLINE:
            break
        r_init = np.random.rand(D) < 0.5
        r_init = r_init.astype(np.float64)
        r_sol, ml_sol = local_search(r_init, max_iters=20000)
        if ml_sol < best_ml_overall - 1e-12:
            best_ml_overall = ml_sol
            best_routing_overall = r_sol.copy()
    
    # --- Phase 2: Branch and bound for exact/near-exact optimization ---
    # With D=15, 2^15=32768 is small. Do a complete B&B.
    def branch_and_bound():
        """Complete branch-and-bound search over all 2^15 assignments."""
        nonlocal best_ml_overall, best_routing_overall
        
        best_ml_bb = best_ml_overall
        best_routing_bb = best_routing_overall.copy()
        
        # Use iterative deepening / DFS with pruning
        # State: (demand_index, current_load, current_routing)
        # Prune if any edge load already exceeds best_ml_bb
        
        def dfs(idx, current_load, current_routing):
            nonlocal best_ml_bb, best_routing_bb
            
            if time.time() > DEADLINE:
                return
            
            # Prune: if any edge already exceeds best, no need to continue
            if np.max(current_load) >= best_ml_bb - 1e-12:
                return
            
            if idx == D:
                ml = np.max(current_load)
                if ml < best_ml_bb - 1e-12:
                    best_ml_bb = ml
                    best_routing_bb = current_routing.copy()
                return
            
            # Branch on demand idx: try CW (0) then CCW (1)
            # Try CW first
            new_load_cw = current_load + cw_w[idx]
            if np.max(new_load_cw) < best_ml_bb - 1e-12:
                current_routing[idx] = 0.0
                dfs(idx + 1, new_load_cw, current_routing)
            
            if time.time() > DEADLINE:
                return
            
            # Try CCW
            new_load_ccw = current_load + ccw_w[idx]
            if np.max(new_load_ccw) < best_ml_bb - 1e-12:
                current_routing[idx] = 1.0
                dfs(idx + 1, new_load_ccw, current_routing)
        
        # Start DFS
        dfs(0, np.zeros(n_edges, dtype=np.float64), np.zeros(D, dtype=np.float64))
        
        return best_ml_bb, best_routing_bb
    
    # Run B&B
    best_ml_overall, best_routing_overall = branch_and_bound()
    
    # --- Phase 3: Final polish with more aggressive local search ---
    if time.time() < DEADLINE - 5:
        # Try more restarts around the best solution
        for restart in range(100):
            if time.time() > DEADLINE:
                break
            # Perturb the best solution
            r_perturbed = best_routing_overall.copy()
            n_flips = np.random.randint(1, 4)
            flip_indices = np.random.choice(D, size=n_flips, replace=False)
            r_perturbed[flip_indices] = 1.0 - r_perturbed[flip_indices]
            r_sol, ml_sol = local_search(r_perturbed, max_iters=50000)
            if ml_sol < best_ml_overall - 1e-12:
                best_ml_overall = ml_sol
                best_routing_overall = r_sol.copy()
    
    # Return the routing
    return best_routing_overall

# EVOLVE-BLOCK-END
