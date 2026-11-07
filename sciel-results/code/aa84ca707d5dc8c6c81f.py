import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for the ACO-CVRP instances.
    
    Strategy: Geometric Angle-Based Heuristic with Demand Awareness.
    Instead of just inverse distance, we penalize sharp turns (high angle between 
    incoming and outgoing edges) to encourage smooth, Euclidean-TSP-like routes.
    We also include a demand-awareness factor to prefer packing high-demand nodes 
    together or early, reducing the number of trips.
    
    eta[i, j] = (1 / d(i, j)^gamma) * (1 + beta * smoothness(i, j)) * (1 / (1 + alpha * demand[j]))
    
    However, 'smoothness' depends on the predecessor node. Since the heuristic 
    matrix is static (computed once), we cannot easily encode a 3D tensor 
    (predecessor, current, next) into a 2D matrix (current, next) without 
    approximation.
    
    A common static approximation for CVRP/TSP heuristics that captures local 
    geometry is to use the "nearest neighbor" or "sweep" angle relative to the 
    depot, or simply rely on the fact that short edges in a clustered problem 
    often imply smooth paths. 
    
    A more robust static heuristic that is known to work well for ACO-CVRP is 
    the "nearest neighbor" distance to the depot, or a combination of distance 
    and demand.
    
    Let's try a different structural approach: 
    Use a **Sweep-Line Greedy Proxy** to evaluate the heuristic parameters.
    The proxy simulates the ants' decision-making process deterministically 
    (greedy choice based on the heuristic) to estimate the total tour length.
    This is much faster than a full ACO run and allows for a finer search 
    over the parameter space (gamma, alpha, beta).
    
    The heuristic will be:
    eta[i, j] = d(i, j)^(-gamma) * (1 / (1 + alpha * demand[j]))
    
    We will search for optimal gamma and alpha.
    """
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)

    def make_eta(gamma, alpha):
        """Compute heuristic matrix for given gamma and alpha."""
        # Inverse distance
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_inv = np.where(D > 0, D ** (-gamma), 1e-9)
        
        # Demand penalty: 1 / (1 + alpha * demand[j])
        # We want to penalize high demand nodes if alpha > 0?
        # Actually, if we want to pack heavy nodes early, we might want to 
        # attract to them? Or repel?
        # In CVRP, the number of trips is driven by how well we pack.
        # A common heuristic is to just use distance.
        # Let's try: eta[i,j] = d(i,j)^-gamma * (1 + alpha * (capacity - demand[j]))
        # This rewards nodes with less demand (easier to fit later)?
        # Or maybe: eta[i,j] = d(i,j)^-gamma / (1 + alpha * demand[j])
        # Let's stick to the prompt's suggestion: penalty for high demand.
        demand_factor = 1.0 / (1.0 + alpha * demands)
        
        eta = dist_inv * demand_factor
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def greedy_proxy(eta, seed=0):
        """
        Run a deterministic greedy tour construction using the heuristic.
        This approximates the behavior of an ant with no pheromone 
        (or uniform pheromone) and is very fast.
        Returns the total distance of the constructed routes.
        """
        # To make it deterministic and reproducible for the same eta,
        # we don't actually need a seed if we use argmax.
        # But the real ACO uses probability. A greedy proxy using argmax(eta)
        # is a good lower-bound estimate of the quality.
        
        total_distance = 0.0
        unvisited = set(range(1, n))
        current = 0
        current_load = 0.0
        
        # We need to simulate the full routing process.
        # Since it's greedy, we pick the best next node.
        
        while unvisited:
            rem_cap = capacity - current_load
            feasible = [j for j in unvisited if demands[j] <= rem_cap]
            
            if not feasible:
                # Return to depot
                total_distance += D[current, 0]
                current = 0
                current_load = 0.0
                continue
            
            # Pick the node with the highest eta value
            # eta[current, j]
            best_j = max(feasible, key=lambda j: eta[current, j])
            
            total_distance += D[current, best_j]
            current_load += demands[best_j]
            unvisited.remove(best_j)
            current = best_j
            
            if not unvisited:
                total_distance += D[current, 0]
                break
        
        return total_distance

    def evaluate_params(gamma, alpha):
        """Evaluate a pair of parameters using the greedy proxy."""
        eta = make_eta(gamma, alpha)
        cost = greedy_proxy(eta)
        return cost

    # Search for optimal gamma and alpha
    best_gamma = 1.0
    best_alpha = 0.0
    best_cost = float('inf')

    # Coarse grid
    gammas = np.arange(0.5, 3.5, 0.25)
    alphas = np.arange(0.0, 2.0, 0.25)

    for g in gammas:
        for a in alphas:
            if time.time() > deadline - 10.0:
                break
            cost = evaluate_params(g, a)
            if cost < best_cost:
                best_cost = cost
                best_gamma = g
                best_alpha = a

    # Fine grid around best
    if time.time() < deadline - 5.0:
        fine_gammas = np.arange(best_gamma - 0.2, best_gamma + 0.2, 0.05)
        fine_alphas = np.arange(max(0, best_alpha - 0.2), best_alpha + 0.2, 0.05)
        
        for g in fine_gammas:
            for a in fine_alphas:
                if time.time() > deadline - 5.0:
                    break
                cost = evaluate_params(g, a)
                if cost < best_cost:
                    best_cost = cost
                    best_gamma = g
                    best_alpha = a

    # Final heuristic
    final_eta = make_eta(best_gamma, best_alpha)
    
    # Safety checks
    if final_eta.shape != D.shape:
        final_eta = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_eta, 1e-9)

    final_eta = np.where(np.isfinite(final_eta) & (final_eta > 0), final_eta, 1e-9)

    return final_eta
