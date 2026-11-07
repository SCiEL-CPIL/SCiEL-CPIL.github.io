import numpy as np
import time

def _run_aco_sim(dist, eta, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation to evaluate a heuristic candidate.
    Uses the exact protocol described in the task contract:
    - 20 ants, 30 iterations
    - Pheromone starts at 1
    - Heuristic non-positive becomes 1e-9
    - Probability proportional to tau * eta
    - Directed pheromone update
    - Evaporation 0.9
    - Reward 1/length
    """
    n = len(dist)
    
    # Sanitize heuristic
    eta_safe = np.where(eta > 0, eta, 1e-9)
    
    # Pheromone init
    tau = np.ones((n, n))
    
    # Mask for diagonal (self-loops not allowed)
    # We will handle visited nodes dynamically
    
    best_len = float('inf')
    rng = np.random.default_rng(seed)
    
    # Precompute log probabilities? No, tau changes.
    # We can precompute the structure of the loop for speed if possible, 
    # but n=50 is small enough that pure Python loop with numpy row ops is okay.
    
    for _ in range(n_iters):
        tours = []
        for _ in range(n_ants):
            tour = [0]
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            curr = 0
            
            for _ in range(n - 1):
                # Calculate probabilities for current node
                # probs[j] = tau[curr, j] * eta[curr, j] if not visited[j]
                
                # Get row
                row_tau = tau[curr, :]
                row_eta = eta_safe[curr, :]
                
                # Mask visited
                # Create a mask where visited is False
                # We need to zero out visited nodes in the probability vector
                # It's faster to compute the product and then mask
                probs = row_tau * row_eta
                probs[visited] = 0.0
                
                total_prob = np.sum(probs)
                
                if total_prob == 0:
                    # Fallback: random unvisited
                    unvisited = np.where(~visited)[0]
                    next_node = rng.choice(unvisited)
                else:
                    # Normalize
                    # Use multinomial or choice with p
                    # To save time, we can use the inverse CDF method manually?
                    # rng.choice is implemented in C, so it's fast.
                    next_node = rng.choice(n, p=probs / total_prob)
                
                tour.append(next_node)
                visited[next_node] = True
                curr = next_node
            
            # Close tour
            tour.append(0)
            
            # Calculate length
            # Vectorized length calculation
            # tour is list of ints
            # dist[tour[i], tour[i+1]]
            # We can do this with numpy indexing
            tour_arr = np.array(tour)
            # Length is sum of dist[tour_arr[i], tour_arr[i+1]]
            # dist[tour_arr[:-1], tour_arr[1:]] gives array of lengths
            lengths = dist[tour_arr[:-1], tour_arr[1:]]
            length = np.sum(lengths)
            
            tours.append((length, tour_arr))
            
            if length < best_len:
                best_len = length
        
        # Update Pheromones
        tau *= 0.9
        
        # Add rewards
        # For each ant, add 1/length to tau[u, v] for each step u->v
        # This is directed
        for length, tour_arr in tours:
            reward = 1.0 / length
            # We need to update tau[tour_arr[i], tour_arr[i+1]] += reward
            # Vectorized update:
            # u = tour_arr[:-1]
            # v = tour_arr[1:]
            # np.add.at(tau, (u, v), reward)
            u = tour_arr[:-1]
            v = tour_arr[1:]
            np.add.at(tau, (u, v), reward)

    return best_len

def heuristics(distance_matrix):
    """
    Compute heuristic matrix based on local neighborhood density.
    Strategy:
    1. Compute k-NN for each node.
    2. Estimate local density (e.g., average distance to k-NNs or count of neighbors within radius).
    3. Construct eta[i, j] = base_heuristic * density_factor[j]
       Where base_heuristic is 1/d(i,j).
       density_factor[j] penalizes nodes that are 'central' or 'crowded' to encourage visiting 'peripheral' nodes,
       or rewards them? 
       
       Let's think: In TSP, a good tour often follows the convex hull and avoids crossing the interior.
       Nodes on the convex hull have lower "density" (fewer neighbors close by in a specific direction, or larger average k-NN distance).
       Interior nodes have higher density (smaller average k-NN distance).
       
       If we make eta[i,j] smaller for j that are interior (high density), the ant is less likely to go to interior nodes.
       This might help in constructing a tour that stays on the perimeter, which is often optimal for uniform random points.
       
       Let's try: eta[i, j] = (1 / d(i,j)) * (1 / (local_density[j] + epsilon))
       Or perhaps: eta[i, j] = (1 / d(i,j)) * (local_density[j] + epsilon)
       
       Let's evaluate both or a mix.
       Actually, standard 1/d is already biased towards short edges.
       If we divide by density, we penalize going to dense areas.
       If we multiply by density, we reward going to dense areas.
       
       For uniform random points, the optimal tour length is proportional to sqrt(N).
       The tour typically visits points in a space-filling manner but avoids long jumps.
       
       Let's try to optimize the parameter beta in:
       eta[i, j] = (1 / d(i,j)) * (avg_knn_dist[j])^beta
       
       If beta > 0, we prefer going to nodes with larger avg_knn_dist (peripheral).
       If beta < 0, we prefer going to nodes with smaller avg_knn_dist (central).
       
       We will search over k and beta.
    """
    start_time = time.time()
    deadline = start_time + 280  # 50s margin
    
    n = len(distance_matrix)
    
    # Precompute k-NN distances for a few k values
    # We need to find k-NN for each point.
    # Using scipy.spatial.cKDTree is efficient.
    from scipy.spatial import cKDTree
    
    # Reconstruct points from distance matrix?
    # The task says "50 uniform random points in the unit square".
    # We are given the distance_matrix.
    # We can use MDS to get approximate points, or just work with distances.
    # However, k-NN is naturally defined on points.
    # Can we compute k-NN from distance matrix? Yes.
    # For each i, sort distance_matrix[i] and take the k smallest (excluding 0).
    
    # Let's compute local density metrics directly from distance matrix.
    # Metric 1: Average distance to k nearest neighbors.
    # Metric 2: Sum of distances to k nearest neighbors.
    
    # Let's try a few k values
    k_values = [5, 10, 15, 20]
    beta_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    # Precompute sorted distances
    # dist_sorted[i] = sorted distance_matrix[i]
    dist_sorted = np.sort(distance_matrix, axis=1)
    # dist_sorted[:, 0] is 0 (self)
    # dist_sorted[:, 1] is nearest neighbor distance
    
    # Precompute base heuristic 1/d
    # Handle 0 on diagonal
    base_eta = np.zeros_like(distance_matrix)
    np.fill_diagonal(distance_matrix, 1.0) # Temporary
    base_eta = 1.0 / distance_matrix
    np.fill_diagonal(base_eta, 0.0) # Diagonal doesn't matter, will be masked
    np.fill_diagonal(distance_matrix, 0.0) # Restore original? We don't modify original, we used a copy?
    # Wait, I modified distance_matrix in the code above? 
    # "np.fill_diagonal(distance_matrix, 1.0)" modifies the input! 
    # I must not modify the input.
    
    # Let's redo this safely.
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, 1.0)
    base_eta = 1.0 / dist_safe
    np.fill_diagonal(base_eta, 0.0)
    
    # Compute density metrics for each k
    density_metrics = {}
    for k in k_values:
        # Average distance to k nearest neighbors (indices 1 to k)
        # dist_sorted[:, 1:k+1]
        avg_knn = np.mean(dist_sorted[:, 1:k+1], axis=1)
        density_metrics[k] = avg_knn
        
    best_score = float('inf')
    best_eta = base_eta.copy()
    
    # Search over k and beta
    # We will run the ACO sim for each combination.
    # To save time, we can do a coarse grid search first.
    
    # Let's flatten the search space
    candidates = []
    for k in k_values:
        for beta in beta_values:
            candidates.append((k, beta))
            
    # Shuffle candidates to avoid bias? Or just iterate.
    # We have ~24 candidates.
    # Each ACO sim takes ~1-2s? 
    # 24 * 2s = 48s. This is well within budget.
    
    # Let's run the search
    for k, beta in candidates:
        if time.time() > deadline:
            break
            
        avg_knn = density_metrics[k]
        
        # Construct eta
        # eta[i, j] = base_eta[i, j] * (avg_knn[j])^beta
        # Note: avg_knn[j] is a scalar for node j.
        # We need to broadcast it to the row i.
        # eta[i, j] depends on j's property.
        
        # Create a matrix where each column j is scaled by avg_knn[j]^beta
        scale = (avg_knn ** beta)
        # scale is shape (n,)
        # We want eta[i, j] = base_eta[i, j] * scale[j]
        # This is base_eta * scale[None, :]
        
        eta_candidate = base_eta * scale[None, :]
        
        # Ensure positive
        eta_candidate = np.where(eta_candidate > 0, eta_candidate, 1e-9)
        
        # Evaluate
        score = _run_aco_sim(distance_matrix, eta_candidate, n_ants=20, n_iters=30, seed=42)
        
        if score < best_score:
            best_score = score
            best_eta = eta_candidate.copy()
            
    # If no candidate was better than base_eta (which is 1/d), best_eta might be base_eta
    # But we initialized best_eta to base_eta.
    # However, if the search found a better one, best_eta is updated.
    
    # Final sanity check
    if not np.all(np.isfinite(best_eta)):
        return base_eta
        
    return best_eta
