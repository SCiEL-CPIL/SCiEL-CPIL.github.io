"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ant colony optimization of CVRP.
    Returns edge-attractiveness matrix (larger = more attractive).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    # Basic heuristic: 1/distance
    # We want to tune this to be more effective for the ACO
    # The key insight: we can run a quick ACO simulation to find a good heuristic
    # that leads to better solutions
    
    # Convert inputs
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Precompute some useful things
    # Distance from depot to each node
    depot_dists = dm[0, :]
    
    # Strategy: Use a combination of:
    # 1. 1/distance (standard)
    # 2. Penalize long distances more strongly
    # 3. Consider demand (higher demand nodes might be harder to place)
    
    # Let's try a power-law heuristic: eta_ij = 1 / (d_ij + epsilon)^gamma
    # With gamma > 1, we penalize long distances more
    # With gamma < 1, we're less sensitive to distance differences
    
    # Also consider: nodes with high demand relative to capacity are "expensive"
    # to serve, so we might want to prioritize visiting them when they're close
    
    # Let's run a quick optimization: try several gamma values and pick the best
    # based on a fast greedy/ACO proxy
    
    def run_quick_proxy(gamma, alpha=1.0, beta=1.0, iters=5, ants=5):
        """Quick proxy ACO to evaluate heuristic quality."""
        eta = 1.0 / (dm + 1e-9) ** gamma
        
        # Ensure positive
        eta[eta <= 0] = 1e-9
        eta[~np.isfinite(eta)] = 1e-9
        
        best_cost = float('inf')
        
        for it in range(iters):
            if time.time() > deadline - 30:
                break
            pher = np.ones((n, n))
            iter_best = float('inf')
            
            for ant in range(ants):
                # Build route
                route = [0]
                current = 0
                load = 0.0
                cost = 0.0
                visited = set([0])
                
                while True:
                    # Find unvisited customers that fit
                    remaining_cap = cap - load
                    candidates = []
                    for j in range(1, n):
                        if j not in visited and dem[j] <= remaining_cap:
                            candidates.append(j)
                    
                    if not candidates:
                        # Return to depot
                        cost += dm[current, 0]
                        route.append(0)
                        break
                    
                    # Choose next with probability proportional to pheromone * heuristic
                    cand_arr = np.array(candidates)
                    probs = pher[current, cand_arr] * eta[current, cand_arr]
                    total = probs.sum()
                    if total <= 0 or not np.isfinite(total):
                        # Fallback: pick closest
                        dists = dm[current, cand_arr]
                        next_node = cand_arr[np.argmin(dists)]
                    else:
                        probs = probs / total
                        next_node = cand_arr[np.random.choice(len(cand_arr), p=probs)]
                    
                    cost += dm[current, next_node]
                    load += dem[next_node]
                    route.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if cost < iter_best:
                    iter_best = cost
            
            if iter_best < best_cost:
                best_cost = iter_best
            
            # Update pheromone
            pher *= 0.9
            pher += 1.0 / max(iter_best, 1e-9)
        
        return best_cost
    
    # Try a few gamma values
    gamma_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    best_gamma = 1.0
    best_proxy_cost = float('inf')
    
    np.random.seed(42)  # Fixed seed for reproducibility in heuristic selection
    
    for g in gamma_candidates:
        if time.time() > deadline - 60:
            break
        proxy_cost = run_quick_proxy(g)
        if proxy_cost < best_proxy_cost:
            best_proxy_cost = proxy_cost
            best_gamma = g
    
    # Build final heuristic with best gamma
    eta = 1.0 / (dm + 1e-9) ** best_gamma
    
    # Ensure all values are positive and finite
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    
    # Also set diagonal to a small value (no self-loops)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
