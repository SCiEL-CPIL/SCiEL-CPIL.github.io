import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO by simulating a 'Ghost Colony' ACO process.
    
    Strategy:
    1. Simulate the ACO dynamics (ants, pheromones, iterations) but with a fast, 
       approximate cost calculation or reduced ant count to fit the time budget.
    2. Instead of returning the final pheromone matrix, we construct a heuristic 
       matrix that biases the real ACO towards the edges that the Ghost Colony 
       reinforced most strongly, especially in the later iterations where the 
       colony has converged.
    3. The heuristic is a combination of inverse distance (base) and the 
       accumulated 'ghost pheromone' intensity, normalized.
    
    This approach encodes the expected search trajectory into the static heuristic,
    guiding the real colony to avoid early exploration costs and jump straight 
    into the promising region of the search space.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    n = len(demands)
    demands = np.array(demands, dtype=float)
    dist = np.array(distance_matrix, dtype=float)
    
    # Base heuristic: Inverse distance
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-6, dist, 1e6)
    base_eta = 1.0 / dist_safe
    
    # Ghost Colony Simulation Parameters
    # We want to simulate the "wisdom" of the colony.
    # Using fewer ants and iterations in the ghost simulation to stay fast,
    # but enough to capture the main trends.
    ghost_ants = 10
    ghost_iters = 15
    rho = 0.9  # Evaporation rate
    alpha = 1.0  # Pheromone importance in ghost sim
    beta = 2.0  # Heuristic importance in ghost sim (higher to converge faster)
    
    # Initialize Ghost Pheromones
    # Start with uniform pheromone
    tau = np.ones((n, n))
    
    # Track the "best" pheromone accumulation or just use the final state?
    # The task says the real ACO uses heuristic[i,j].
    # We want heuristic[i,j] to be high where the ghost colony found good paths.
    # Let's accumulate the pheromone deposits weighted by iteration (later iterations matter more).
    accumulated_tau = np.zeros((n, n))
    
    # Helper to solve a single ant's path in the ghost colony
    def ghost_ant_path(rng):
        current = 0
        unvisited = set(range(1, n))
        load = 0.0
        total_dist = 0.0
        path_edges = []
        
        while unvisited:
            # Filter feasible customers
            feasible = []
            for c in unvisited:
                if load + demands[c] <= capacity:
                    feasible.append(c)
            
            if not feasible:
                # Return to depot
                total_dist += dist[current, 0]
                path_edges.append((current, 0))
                current = 0
                load = 0.0
                continue
            
            # Calculate probabilities: tau^alpha * eta^beta
            # tau is current ghost pheromone
            # eta is base_eta
            probs = np.zeros(len(feasible))
            for i, c in enumerate(feasible):
                probs[i] = (tau[current, c] ** alpha) * (base_eta[current, c] ** beta)
            
            # Normalize
            sum_probs = np.sum(probs)
            if sum_probs == 0:
                # Fallback to random
                idx = rng.integers(0, len(feasible))
            else:
                probs = probs / sum_probs
                idx = rng.choice(len(feasible), p=probs)
            
            next_c = feasible[idx]
            total_dist += dist[current, next_c]
            path_edges.append((current, next_c))
            
            unvisited.remove(next_c)
            current = next_c
            load += demands[next_c]
        
        # Return to depot
        if current != 0:
            total_dist += dist[current, 0]
            path_edges.append((current, 0))
            
        return total_dist, path_edges

    rng = np.random.default_rng(12345) # Fixed seed for ghost sim determinism
    
    # Run Ghost Colony
    for it in range(ghost_iters):
        if time.time() > deadline - 10.0:
            break
            
        # Reset best cost for this iteration
        best_cost = float('inf')
        best_path_edges = None
        all_paths = []
        
        for _ in range(ghost_ants):
            cost, edges = ghost_ant_path(rng)
            all_paths.append((cost, edges))
            if cost < best_cost:
                best_cost = cost
                best_path_edges = edges
        
        # Update Pheromones
        # Evaporation
        tau = tau * rho
        
        # Deposit
        # Standard ACO: each ant deposits Q/cost on its path.
        # Here we just use 1/cost for simplicity, scaled.
        # To make the heuristic stronger, we might want to emphasize the best path more.
        # But standard ACO behavior is what we are simulating.
        
        # We want to accumulate the "strength" of edges.
        # Let's add the pheromone deposited in this iteration to accumulated_tau.
        # Weight later iterations more heavily?
        # The real ACO runs 30 iterations. The ghost runs 15.
        # If we assume the ghost converges faster due to higher beta, the final state is good.
        
        iter_tau_update = np.zeros((n, n))
        for cost, edges in all_paths:
            if cost > 0:
                deposit = 1.0 / cost
                for u, v in edges:
                    iter_tau_update[u, v] += deposit
        
        tau += iter_tau_update
        
        # Accumulate for final heuristic
        # We want the heuristic to reflect the "consensus" of the ghost colony.
        # Simple sum of tau over time, or just final tau?
        # Final tau is a snapshot. Sum might be better to capture robustness.
        # Let's use the final tau, but normalized.
        # Actually, let's accumulate a weighted sum where later iterations count more.
        weight = it + 1
        accumulated_tau += weight * tau
        
    # Construct Final Heuristic
    # The heuristic should be proportional to the attractiveness.
    # We combine the base inverse distance with the ghost pheromone information.
    # eta_final = base_eta * (1 + gamma * normalized_tau)
    
    # Normalize accumulated_tau
    max_tau = np.max(accumulated_tau)
    if max_tau > 0:
        norm_tau = accumulated_tau / max_tau
    else:
        norm_tau = np.zeros_like(accumulated_tau)
    
    # Symmetrize tau since distance is likely symmetric and problem is undirected
    norm_tau = (norm_tau + norm_tau.T) / 2.0
    base_eta = (base_eta + base_eta.T) / 2.0
    
    # Combine
    # If norm_tau is 0 for an edge, it means the ghost colony never used it.
    # We still want a small attractiveness based on distance.
    # If norm_tau is high, we want high attractiveness.
    
    gamma = 10.0  # Boost factor for ghost pheromone
    result = base_eta * (1.0 + gamma * norm_tau)
    
    # Ensure positive and finite
    result[~np.isfinite(result)] = 1e-9
    result[result < 1e-9] = 1e-9
    
    # Final sanity check: shape and values
    if result.shape != (n, n):
        raise ValueError("Wrong shape")
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite values")
        
    return result
