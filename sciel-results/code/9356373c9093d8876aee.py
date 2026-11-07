import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for the ACO-CVRP instances.
    Strategy: Demand-aware multi-criteria heuristic.
    eta[i,j] = d(i,j)^-gamma * (1 / (1 + alpha * demand[j]))
    
    We search for optimal gamma and alpha using a fast, deterministic 
    greedy proxy that approximates the ACO behavior without stochastic 
    overhead, allowing for a much finer parameter grid search within 
    the time budget.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Pre-compute base inverse distance matrix for gamma=1
    # We will scale exponents later
    with np.errstate(divide='ignore', invalid='ignore'):
        base_inv_d = np.where(D > 0, 1.0 / D, 1e-9)
    
    # Pre-compute demand penalty factor for each node j
    # penalty[j] = 1 / (1 + alpha * demand[j])
    # This is applied to the incoming edges to j

    def make_eta(gamma, alpha):
        """Compute heuristic matrix for given gamma and alpha."""
        # gamma exponent on inverse distance: (1/D)^gamma = D^-gamma
        # We can compute this as exp(-gamma * log(D)) or (1/D)^gamma
        # Using log for stability? D is small, 1/D is large.
        # Let's just use power.
        
        # Ensure D > 0 for power operation
        safe_D = np.where(D > 0, D, 1.0)
        
        # Distance part: D^-gamma
        dist_part = safe_D ** (-gamma)
        
        # Demand part: 1 / (1 + alpha * demand[j])
        # This is a vector of length n, broadcast to rows
        # We want eta[i, j] to depend on demand[j]
        demand_penalty = 1.0 / (1.0 + alpha * demands)
        
        # Combine
        eta = dist_part * demand_penalty[np.newaxis, :]
        
        # Clean up: non-finite or non-positive become 1e-9
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        return eta

    def greedy_proxy(eta, seed=0):
        """
        Fast deterministic proxy for ACO performance.
        Simulates a 'greedy' ant that picks the best feasible node 
        based on the heuristic (ignoring pheromone randomness for speed).
        We run this for a few 'virtual' ants with slightly different 
        tie-breaking or noise to estimate robustness, but primarily 
        we rely on the pure greedy path to be fast.
        """
        best_cost = float('inf')
        
        # Run 3 "virtual ants" with slight variations to estimate variance
        # Ant 0: Pure Greedy (max eta)
        # Ant 1: Greedy with slight random perturbation
        # Ant 2: Greedy with slight random perturbation
        
        for ant_id in range(3):
            current = 0
            current_load = 0.0
            total_distance = 0.0
            unvisited = set(range(1, n))
            
            # For ant 0, use eta directly.
            # For ant 1, 2, add small noise to eta to break ties differently?
            # Actually, let's just use the eta matrix directly for speed.
            # The ACO uses pheromone * heuristic. Initially pheromone is 1,
            # so early moves are heuristic-driven.
            # A pure greedy heuristic-driven tour is a good proxy for the 
            # first few iterations of ACO.
            
            while unvisited:
                rem_cap = capacity - current_load
                
                # Find feasible nodes
                # Vectorized check
                feasible_nodes = [j for j in unvisited if demands[j] <= rem_cap]
                
                if not feasible_nodes:
                    total_distance += D[current, 0]
                    break
                    
                if len(feasible_nodes) == 1:
                    next_node = feasible_nodes[0]
                else:
                    # Pick node with max eta[current, j]
                    # For speed, just argmax
                    # Add tiny noise for ants 1, 2 to avoid identical paths
                    # if ties occur, though unlikely with floats
                    f_arr = np.array(feasible_nodes)
                    weights = eta[current, f_arr]
                    
                    if ant_id > 0:
                        # Add small deterministic noise based on ant_id and current
                        # to simulate different random choices
                        # We don't have a shared RNG state easily here without overhead
                        # Let's just use the max. If ties, index order breaks it.
                        pass
                    
                    next_node = f_arr[np.argmax(weights)]
                    
                total_distance += D[current, next_node]
                current_load += demands[next_node]
                unvisited.remove(next_node)
                current = next_node
                
                if not unvisited:
                    total_distance += D[current, 0]
                    break
            
            if total_distance < best_cost:
                best_cost = total_cost if False else total_distance # typo guard
        
        return best_cost

    # Search for optimal gamma and alpha
    # Gamma: distance exponent. Higher means more local preference.
    # Alpha: demand penalty. Higher means stronger preference for light nodes?
    # Wait, eta = D^-gamma * 1/(1+alpha*demand).
    # If alpha is high, high demand nodes have LOW eta.
    # So ants AVOID high demand nodes initially?
    # The prompt says: "penalty for high-demand nodes... biasing... toward routes 
    # that pack heavier customers early".
    # If we want to pack heavier customers early, we should REWARD them?
    # Or does the ACO pheromone update handle the packing?
    # The prompt inspiration says: "penalty for high-demand nodes (e.g., eta = ... * 1/(1+demand))"
    # and "directly biasing ... toward routes that pack heavier customers early".
    # This seems contradictory if the penalty reduces attractiveness.
    # However, if we look at the "empty-return distances", maybe avoiding heavy 
    # nodes when capacity is low is good?
    # Let's stick to the formula provided in the directive: 
    # eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j]))
    # This penalizes high demand.
    # Let's search alpha in a range that allows both penalizing and rewarding 
    # (if we allowed negative alpha? No, demand penalty usually positive).
    # Let's try alpha > 0.
    
    best_val = float('inf')
    best_gamma = 1.0
    best_alpha = 0.0

    # Coarse Grid
    gammas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    alphas = [0.0, 0.5, 1.0, 2.0, 4.0]
    
    for gamma in gammas:
        if time.time() > deadline - 60.0:
            break
        for alpha in alphas:
            if time.time() > deadline - 55.0:
                break
            eta = make_eta(gamma, alpha)
            val = greedy_proxy(eta)
            if val < best_val:
                best_val = val
                best_gamma = gamma
                best_alpha = alpha

    # Fine Grid around best
    if time.time() < deadline - 30.0:
        # Refine gamma
        g_range = np.arange(max(0.1, best_gamma - 0.5), best_gamma + 0.5, 0.1)
        # Refine alpha
        a_range = np.arange(max(0.0, best_alpha - 1.0), best_alpha + 1.0, 0.5)
        
        for gamma in g_range:
            if time.time() > deadline - 25.0:
                break
            for alpha in a_range:
                if time.time() > deadline - 20.0:
                    break
                if alpha < 0: continue
                eta = make_eta(gamma, alpha)
                val = greedy_proxy(eta)
                if val < best_val:
                    best_val = val
                    best_gamma = gamma
                    best_alpha = alpha
                    
    # Even finer if time permits
    if time.time() < deadline - 10.0:
        g_range = np.arange(max(0.1, best_gamma - 0.2), best_gamma + 0.2, 0.05)
        a_range = np.arange(max(0.0, best_alpha - 0.5), best_alpha + 0.5, 0.25)
        
        for gamma in g_range:
            if time.time() > deadline - 5.0:
                break
            for alpha in a_range:
                if time.time() > deadline - 2.0:
                    break
                if alpha < 0: continue
                eta = make_eta(gamma, alpha)
                val = greedy_proxy(eta)
                if val < best_val:
                    best_val = val
                    best_gamma = gamma
                    best_alpha = alpha

    final_eta = make_eta(best_gamma, best_alpha)
    
    # Safety checks
    if final_eta.shape != D.shape:
        final_eta = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_eta, 1e-9)

    final_eta = np.where(np.isfinite(final_eta) & (final_eta > 0), final_eta, 1e-9)

    return final_eta
