"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _proxy_aco(distance_matrix, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42, deadline=None):
    """
    Run a proxy ACO to evaluate a heuristic's quality.
    This is a lightweight version of the actual solver used to tune the heuristic parameters.
    """
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]
    dm = distance_matrix
    eta = heuristic
    tau = np.ones((n, n))
    best_cost = np.inf
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        iter_best = np.inf
        for _ in range(n_ants):
            current = 0
            load = 0.0
            visited = set()
            route_cost = 0.0
            while True:
                # Find feasible unvisited customers
                feasible = []
                # Optimization: Pre-compute feasible list? No, load changes.
                for j in range(1, n):
                    if j not in visited and demands_arr[j] <= capacity - load:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    route_cost += dm[current, 0]
                    break
                
                # Compute probabilities
                # Vectorized probability calculation for speed
                # Note: We only care about feasible nodes
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = tau[current, j] * eta[current, j]
                
                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    probs = np.ones(len(feasible))
                    total = probs.sum()
                probs /= total
                
                # Select next node
                r = rng.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(feasible):
                    idx = len(feasible) - 1
                next_node = feasible[idx]
                
                route_cost += dm[current, next_node]
                load += demands_arr[next_node]
                visited.add(next_node)
                current = next_node
            
            if route_cost < iter_best:
                iter_best = route_cost
        
        if iter_best < best_cost:
            best_cost = iter_best
        
        # Update pheromone
        tau *= 0.9
        tau += 1.0 / max(iter_best, 1e-9)
    
    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO by finding optimal parameters via proxy ACO.
    Structural change: Multi-criteria heuristic combining distance, demand penalty, and depot penalty.
    eta[i,j] = (d_ij)^(-gamma) * (1 + demand_j)^(-beta) * depot_penalty
    """
    start_time = time.time()
    # Safety margin: Evaluator kills at 330s. We aim to finish well before.
    # The proxy ACO is the bottleneck.
    deadline = start_time + 300.0
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    # Grid search parameters
    # gamma: distance exponent
    # beta: demand penalty exponent
    # depot_factor: multiplier for edges returning to depot (j=0, i!=0)
    
    # We will search a small grid to keep time low.
    # If time is tight, we fall back to a good default.
    
    gammas = [1.5, 2.0, 2.5]
    betas = [0.0, 0.5, 1.0]
    depot_factors = [0.8, 1.0, 1.2] # 1.2 means *rewarding* return? No, penalty usually <1. 
    # Wait, if we penalize return, we want eta to be LOW for return. 
    # So depot_factor should be < 1 to discourage return.
    # Current program used 0.8.
    # Let's try [0.5, 0.8, 1.0]
    
    depot_factors = [0.5, 0.8, 1.0]
    
    best_score = np.inf
    best_params = (2.0, 0.5, 0.8) # Default fallback
    
    # Pre-compute base matrices where possible? 
    # d_ij is fixed.
    # (1+d_j) is fixed per column j.
    
    d_safe = dm + 1e-9
    
    # Iterate through combinations
    # Total combos: 3 * 3 * 3 = 27.
    # Each proxy ACO run takes ~ (20 ants * 30 iters * 50 nodes avg) operations.
    # 27 * ~30000 ops * constant. Should be fast enough in Python if vectorized?
    # The proxy ACO loop is Python-heavy. 
    # 27 runs might be too slow if each takes > 5s.
    # Let's reduce the grid or optimize the proxy.
    
    # Optimization: Use fewer ants/iters for proxy?
    # Inspirations used full 20/30 and scored 1.4125.
    # Let's try 10 ants, 15 iters for proxy to speed up, then refine best 3 with full?
    # Or just a smaller grid.
    
    # Let's try a coarser grid first.
    gammas = [1.5, 2.0, 2.5]
    betas = [0.0, 1.0]
    depot_factors = [0.8, 1.0]
    
    # 3 * 2 * 2 = 12 runs.
    
    for gamma in gammas:
        for beta in betas:
            for depot_factor in depot_factors:
                if time.time() > deadline - 10.0:
                    break
                
                # Construct heuristic
                # eta = d^-gamma * (1+d)^-beta * depot_mask
                with np.errstate(divide='ignore', invalid='ignore'):
                    eta_dist = np.power(d_safe, -gamma)
                    
                    # Demand term: column-wise
                    # (1 + demand[j])^-beta
                    demand_term = np.power(1.0 + demands_arr, -beta)
                    # eta_demand is a vector, need to broadcast to matrix
                    # eta_demand_matrix[i, j] = demand_term[j]
                    eta_demand = np.tile(demand_term, (n, 1))
                    
                    eta = eta_dist * eta_demand
                    
                    # Apply depot penalty
                    # j=0 is depot. i!=0.
                    # eta[i, 0] *= depot_factor
                    if depot_factor != 1.0:
                        eta[1:, 0] *= depot_factor
                    
                    # Ensure positive and finite
                    eta[~np.isfinite(eta)] = 1e-9
                    eta[eta <= 0] = 1e-9
                    
                    # Run proxy ACO
                    # Use a shorter deadline for the proxy itself to ensure we don't overrun
                    proxy_deadline = deadline - 5.0
                    score = _proxy_aco(dm, demands, capacity, eta, 
                                       n_ants=20, n_iters=30, seed=42, 
                                       deadline=proxy_deadline)
                    
                    if score < best_score:
                        best_score = score
                        best_params = (gamma, beta, depot_factor)
                
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
        if time.time() > deadline - 10.0:
            break
    
    # Construct final heuristic with best params
    gamma, beta, depot_factor = best_params
    
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_dist = np.power(d_safe, -gamma)
        demand_term = np.power(1.0 + demands_arr, -beta)
        eta_demand = np.tile(demand_term, (n, 1))
        heuristic = eta_dist * eta_demand
        
        if depot_factor != 1.0:
            heuristic[1:, 0] *= depot_factor
    
    # Final safety checks
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Set diagonal to small value (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
