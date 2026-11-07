import numpy as np
import time

def _run_aco_sim(dist, eta, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation to evaluate a heuristic candidate.
    Returns the best tour length found.
    """
    n = len(dist)
    # Pheromone init
    tau = np.ones((n, n))
    
    # Local random generator for reproducibility and thread safety (if needed)
    rng = np.random.default_rng(seed)
    
    best_len = float('inf')
    
    for _ in range(n_iters):
        tours = []
        for _ in range(n_ants):
            tour = [0]
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            curr = 0
            
            for _ in range(n - 1):
                # Calculate probabilities
                # Prob proportional to tau[curr, :] * eta[curr, :]
                # Mask visited
                probs = tau[curr, :].copy()
                probs *= eta[curr, :]
                probs[visited] = 0
                
                # Handle if all 0
                total_prob = np.sum(probs)
                if total_prob == 0:
                    # Pick random unvisited
                    unvisited = np.where(~visited)[0]
                    next_node = rng.choice(unvisited)
                else:
                    probs /= total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour.append(next_node)
                visited[next_node] = True
                curr = next_node
            
            # Close tour
            tour.append(0)
            
            # Calculate length
            # Vectorized distance lookup
            tour_arr = np.array(tour)
            length = np.sum(dist[tour_arr[:-1], tour_arr[1:]])
            tours.append((length, tour_arr))
            
            if length < best_len:
                best_len = length
        
        # Update Pheromones
        tau *= 0.9
        for length, tour in tours:
            reward = 1.0 / length
            # Deposit in travel direction only
            # tour is [0, ..., 0]
            # Edges are (tour[i], tour[i+1])
            u = tour[:-1]
            v = tour[1:]
            # Use advanced indexing to add reward to specific cells
            # np.add.at is needed if there are duplicate edges in a single tour? 
            # In a TSP tour, edges are unique per tour, but multiple ants might deposit on same edge.
            # Here we are looping over ants, so we just add.
            # However, u and v are arrays. We can do:
            # tau[u, v] += reward
            # This works if (u,v) pairs are unique within this loop iteration (which they are for a single tour).
            tau[u, v] += reward

    return best_len

def heuristics(distance_matrix):
    """
    Search for the best heuristic exponent alpha in 1/d^alpha.
    Uses a high-fidelity internal ACO simulation to select the most robust alpha.
    """
    start_time = time.time()
    timeout = 280  # Leave 20s margin
    
    n = len(distance_matrix)
    
    # Calculate base inverse distances
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    
    # Candidate alphas
    # We search a range of alphas. 
    # Too low: weak bias. Too high: very strong bias to nearest neighbors, might get stuck in local optima.
    alphas = np.linspace(0.5, 5.0, 21)
    
    # Seeds for robustness evaluation
    seeds = [1, 2, 3, 4, 5]
    
    best_alpha = 1.0
    best_score = float('inf')
    best_eta = None
    
    for alpha in alphas:
        if time.time() - start_time > timeout:
            break
            
        eta = 1.0 / (safe_dist ** alpha)
        # Ensure diagonal is small/positive
        np.fill_diagonal(eta, 1e-9)
        
        # Evaluate this candidate with multiple seeds
        total_score = 0.0
        for s in seeds:
            # Run simulation
            score = _run_aco_sim(distance_matrix, eta, n_ants=20, n_iters=30, seed=s)
            total_score += score
            
            # Check time inside the seed loop too, just in case
            if time.time() - start_time > timeout:
                break
        
        avg_score = total_score / len(seeds)
        
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = alpha
            best_eta = eta.copy()
            
    # Fallback if something went wrong
    if best_eta is None:
        eta = 1.0 / safe_dist
        np.fill_diagonal(eta, 1e-9)
        best_eta = eta

    return best_eta
