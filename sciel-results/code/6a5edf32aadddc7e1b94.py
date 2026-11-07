import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Structural change: Instead of searching for static parameters (beta, alpha),
    this heuristic simulates the expected pheromone accumulation trajectory of
    the ACO algorithm. It assumes a 'greedy-like' successful path and computes
    how pheromone would reinforce edges over time, using this as the heuristic.
    This encodes the dynamic convergence bias directly into the static matrix.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]
    
    start_time = time.time()
    DEADLINE = 300.0  # Safety margin before 330s kill

    # --- Step 1: Build a strong baseline heuristic (Greedy/Savings hybrid) ---
    # We use a blend of inverse distance and a demand-aware penalty.
    # This serves as the initial 'attractiveness' for the trajectory simulation.
    
    # Base: Inverse distance (stronger for closer nodes)
    dist_safe = np.maximum(dm, 1e-9)
    base_eta = 1.0 / dist_safe
    
    # Demand penalty: Avoid edges to nodes with very high demand if capacity is tight?
    # Actually, in CVRP, you want to fill the truck. But for the heuristic,
    # high demand nodes are 'heavy'. 
    # Let's stick to a simple demand-aware factor: prefer moderate demand.
    # However, the most important factor for CVRP is distance.
    # We will use a demand-weighted distance: 1 / (dist * (1 + 0.1*dem[j]))
    # This penalizes going to high-demand nodes slightly, encouraging packing of similar sizes?
    # Or simply: 1/dist is usually robust. Let's add a small depot-proximity bias.
    
    # Depot proximity bias: Nodes close to depot are easier to return from.
    depot_dist = dm[:, 0].copy()
    max_dd = np.max(depot_dist)
    if max_dd > 0:
        norm_dd = depot_dist / max_dd
    else:
        norm_dd = np.zeros(n)
    
    # Bias factor: 1.0 to 1.2 based on proximity to depot
    depot_bias = 1.0 + 0.2 * (1.0 - norm_dd)
    
    # Initial heuristic
    initial_eta = base_eta * depot_bias[:, np.newaxis]
    
    # Demand adjustment: Penalize edges to high demand nodes slightly to encourage
    # grouping? No, usually you want to visit high demand nodes when you have capacity.
    # A static heuristic doesn't know capacity state.
    # Let's keep initial_eta simple: 1/dist * depot_bias.
    
    # --- Step 2: Simulate Pheromone Trajectory ---
    # The ACO algorithm updates pheromone as:
    # tau_new = rho * tau_old + sum(delta_k)
    # where delta_k = 1/cost_k for ant k.
    # In the early iterations, pheromone is uniform (1.0).
    # As iterations proceed, edges on good tours get reinforced.
    # We simulate this process to create a 'target' pheromone matrix that
    # represents the converged state of a successful search.
    
    # We run a fast, deterministic 'proxy' ACO to estimate the final pheromone.
    # To make it fast, we use a small number of ants and iterations, but we
    # simulate the pheromone update exactly as in the main algorithm.
    
    # Parameters for the proxy simulation
    proxy_ants = 5
    proxy_iters = 20
    rho = 0.9  # Evaporation rate from task description
    
    # Initialize pheromone to 1.0 (as per task: "pheromone starts at 1")
    tau = np.ones_like(dm, dtype=np.float64)
    
    # We will run a proxy ACO using the initial_eta to guide the ants.
    # The resulting tau matrix after proxy_iters will be our heuristic.
    # Why? Because the main ACO starts with tau=1 and eta=heuristic.
    # If our heuristic is the 'expected' tau after convergence, it provides
    # a strong directional bias that aligns with the algorithm's own dynamics.
    
    # However, running a full ACO inside the heuristic function might be slow.
    # 50 nodes, 5 ants, 20 iters = 500 paths. Each path is O(n^2) = 2500 ops.
    # 500 * 2500 = 1.25M ops. This is very fast in Python/NumPy if vectorized,
    # but the ant construction is sequential.
    # Let's optimize the proxy ACO construction.
    
    def run_proxy_aco(eta, n_ants, n_iters, dm, dem, capacity, rho):
        """
        Runs a simplified ACO to estimate the converged pheromone matrix.
        Uses greedy selection with a small amount of randomness or pure greedy
        to ensure speed and stability.
        """
        tau = np.ones_like(dm, dtype=np.float64)
        n = dm.shape[0]
        
        for _iter in range(n_iters):
            iteration_best_cost = float('inf')
            # Accumulate pheromone deposits
            deposit = np.zeros_like(dm, dtype=np.float64)
            
            for _ant in range(n_ants):
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                current = 0
                load = 0.0
                total_dist = 0.0
                path_edges = []
                
                for _step in range(n - 1):
                    # Find valid moves
                    valid_moves = []
                    for j in range(1, n):
                        if not visited[j] and load + dem[j] <= capacity:
                            valid_moves.append(j)
                    
                    if not valid_moves:
                        # Return to depot
                        total_dist += dm[current, 0]
                        path_edges.append((current, 0))
                        current = 0
                        load = 0.0
                        continue
                    
                    # Calculate probabilities: tau[i,j] * eta[i,j]
                    # To speed up, we can use a greedy choice with a small epsilon
                    # or pure greedy if we want speed.
                    # Let's use a greedy choice weighted by tau*eta to mimic ACO.
                    # But for speed, pure greedy on (tau*eta) is a good approximation
                    # of the 'best' path the colony would find.
                    
                    best_score = -1.0
                    best_j = -1
                    for j in valid_moves:
                        score = tau[current, j] * eta[current, j]
                        if score > best_score:
                            best_score = score
                            best_j = j
                    
                    if best_j == -1:
                        # Fallback to nearest
                        best_j = valid_moves[0]
                    
                    total_dist += dm[current, best_j]
                    load += dem[best_j]
                    visited[best_j] = True
                    path_edges.append((current, best_j))
                    current = best_j
                
                if current != 0:
                    total_dist += dm[current, 0]
                    path_edges.append((current, 0))
                
                # Record best cost
                if total_dist < iteration_best_cost:
                    iteration_best_cost = total_dist
                
                # Deposit pheromone on all edges in this ant's path
                # delta = 1/cost
                if total_dist > 0:
                    delta = 1.0 / total_dist
                    for u, v in path_edges:
                        deposit[u, v] += delta
                        deposit[v, u] += delta  # Symmetric? Distance matrix is likely symmetric.
                        # Task says distance_matrix, usually symmetric. 
                        # Pheromone update in ACO is usually on the edges traversed.
                        # If the graph is directed, we only add to [u,v]. 
                        # Given "distance matrix", it's likely symmetric Euclidean.
                        # Let's assume symmetric for speed, or just add to [u,v] and [v,u].
                        # The task says "pheromone[i, j]", implying a matrix.
                        # Standard ACO on symmetric TSP/CVRP updates both or treats it as undirected.
                        # Let's update both to be safe.
            
            # Update pheromone
            # tau = rho * tau + deposit
            tau = rho * tau + deposit
            
            # Check time
            if time.time() - start_time > DEADLINE - 50:
                break
        
        return tau

    # Run the proxy ACO
    # We use the initial_eta to guide the proxy.
    # The resulting tau is our final heuristic.
    
    # Optimization: The proxy ACO above is O(n_ants * n_iters * n^2).
    # 5 * 20 * 2500 = 250,000 operations. Very fast.
    
    final_heuristic = run_proxy_aco(initial_eta, proxy_ants, proxy_iters, dm, dem, capacity, rho)
    
    # Safety checks
    if not np.all(np.isfinite(final_heuristic)):
        final_heuristic = initial_eta
    
    # Ensure positive values
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    # Optional: Normalize to avoid extreme values?
    # The ACO uses tau * eta. If eta is large, it dominates.
    # The task says "Heuristic entries that are not positive become 1e-9".
    # It doesn't say to normalize.
    # However, if the heuristic values are too large, they might overflow or dominate
    # the pheromone in the actual run.
    # Let's scale the heuristic to be comparable to the initial pheromone (1.0).
    # We can normalize by the max value, or just leave it.
    # In the actual ACO, pheromone starts at 1. If heuristic is 100, then
    # tau*eta = 100. As pheromone grows, it might become huge.
    # But the task says "pheromone starts at 1".
    # Let's normalize the heuristic to have a max of 1.0 to keep the scale consistent.
    
    max_val = np.max(final_heuristic)
    if max_val > 0:
        final_heuristic = final_heuristic / max_val
    
    # Ensure minimum value
    final_heuristic[final_heuristic < 1e-9] = 1e-9
    
    return final_heuristic
