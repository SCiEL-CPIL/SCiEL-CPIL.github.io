import numpy as np
import time
import random

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Simulates the ACO process exactly as described in the task contract.
    Returns the best total prize found by any ant in any iteration.
    """
    n = len(prizes)
    # Heuristic entries that are not positive become 1e-9
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    # Pheromone starts at 1. Since it updates uniformly (multiply by 0.9, add constant),
    # and all ants start at 0, the pheromone matrix remains uniform across all entries 
    # throughout the simulation. Therefore, the choice probability is proportional 
    # solely to the heuristic H[i, j]. We can optimize this by not explicitly 
    # tracking the pheromone matrix state if we know it's uniform, but to be safe 
    # and strictly follow the "proportional to pheromone * heuristic" rule, 
    # we note that if P is uniform, P[i,j] * H[i,j] is proportional to H[i,j].
    
    for it in range(num_iters):
        best_iter_prize = 0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0
            total_prize = prizes[0]
            
            while True:
                # Candidates are unvisited nodes j such that path_len + d[i,j] + d[j,0] <= max_len
                remaining_budget = max_len - path_len
                # d[i, :] + d[:, 0] must be <= remaining_budget
                # Note: distance_matrix[current_node, :] is d[i, j]
                # distance_matrix[:, 0] is d[j, 0]
                
                # Vectorized check
                # We need to check for all unvisited j
                unvisited = np.where(~visited_mask)[0]
                if len(unvisited) == 0:
                    break
                
                # Calculate cost to go to j and return to 0
                # cost[j] = d[i, j] + d[j, 0]
                costs = distance_matrix[current_node, unvisited] + distance_matrix[unvisited, 0]
                
                # Filter valid candidates
                valid_local_indices = np.where(costs <= remaining_budget)[0]
                
                if len(valid_local_indices) == 0:
                    break
                
                valid_indices = unvisited[valid_local_indices]
                
                # Get probabilities for valid indices
                # Probability is proportional to pheromone[i, j] * heuristic[i, j]
                # Since pheromone is uniform, this is proportional to heuristic[i, j]
                probs = H[current_node, valid_indices]
                total_prob = np.sum(probs)
                
                if total_prob == 0:
                    break
                
                probs = probs / total_prob
                
                # Choose next node
                next_idx_in_valid = rng.choice(len(valid_indices), p=probs)
                next_node = valid_indices[next_idx_in_valid]
                
                # Update state
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        # Pheromone update: P = 0.9 * P + best_iter_prize / 10
        # As established, this keeps P uniform. 
        # However, strictly speaking, if we were tracking P, we would update it here.
        # Since we know P is uniform, we skip the update to save time, 
        # as it does not change the relative probabilities.
        
        # Track best prize in this iteration for potential logging, though not used for P update in this optimization
        # The contract says "every entry gains (best prize of that iteration) / 10".
        # Since we don't use P in the next step's probability calculation (because it's uniform), 
        # we don't need to compute best_iter_prize unless we want to be pedantic.
        # But wait, does "best prize of that iteration" affect anything else? No.
        # So we can skip calculating best_iter_prize entirely.

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    n = len(prizes)
    eps = 1e-9
    
    # Strategy: Greedy Path Pre-computation
    # 1. Run a few greedy heuristics to find good paths.
    # 2. Identify nodes and edges frequently used in these good paths.
    # 3. Construct a heuristic that boosts these specific edges.
    
    # Helper to run a single greedy path with a specific scoring function
    def run_greedy(scoring_func, seed):
        rng = random.Random(seed)
        current_node = 0
        visited = [False] * n
        visited[0] = True
        path_len = 0
        total_prize = prizes[0]
        path = [0]
        
        while True:
            best_score = -1
            best_node = -1
            remaining = max_len - path_len
            
            for j in range(n):
                if not visited[j]:
                    cost = distance_matrix[current_node, j] + distance_matrix[j, 0]
                    if cost <= remaining:
                        score = scoring_func(current_node, j)
                        if score > best_score:
                            best_score = score
                            best_node = j
            
            if best_node == -1:
                break
            
            path_len += distance_matrix[current_node, best_node]
            total_prize += prizes[best_node]
            visited[best_node] = True
            current_node = best_node
            path.append(current_node)
            
        return total_prize, path

    # Define several greedy scoring functions to find diverse good paths
    # 1. Standard prize/distance
    def score_prize_dist(i, j):
        d = distance_matrix[i, j]
        if d < eps: return 0
        return prizes[j] / d
        
    # 2. Prize / (distance + return distance)
    def score_prize_total_dist(i, j):
        d_total = distance_matrix[i, j] + distance_matrix[j, 0]
        if d_total < eps: return 0
        return prizes[j] / d_total
        
    # 3. Prize squared / distance
    def score_prize_sq_dist(i, j):
        d = distance_matrix[i, j]
        if d < eps: return 0
        return (prizes[j] ** 2) / d
        
    # 4. Prize / distance^2
    def score_prize_dist_sq(i, j):
        d = distance_matrix[i, j]
        if d < eps: return 0
        return prizes[j] / (d * d)
        
    # 5. Randomized greedy (add noise to score to explore)
    def make_randomized_greedy(base_score, noise_scale, seed):
        def score(i, j):
            s = base_score(i, j)
            # Add noise
            # Use a simple hash-based pseudo-random for reproducibility in this inner function
            # Or just use the global rng? No, let's keep it deterministic per seed
            # We'll just use a fixed offset based on seed and node index
            noise = (seed * 1000 + i * 10 + j) % 100 / 100.0 * noise_scale
            return s + noise
        return score

    # Run multiple greedy instances to collect good paths
    good_paths = []
    seeds_to_try = [42, 43, 44, 45, 46, 100, 200, 300, 400, 500]
    
    scoring_funcs = [
        score_prize_dist,
        score_prize_total_dist,
        score_prize_sq_dist,
        score_prize_dist_sq,
    ]
    
    # Run deterministic greedies
    for sf in scoring_funcs:
        if time.time() > deadline:
            break
        prize, path = run_greedy(sf, 0)
        if len(path) > 1:
            good_paths.append((prize, path))
            
    # Run randomized greedies
    for sf in scoring_funcs[:2]: # Just the top 2 to save time
        for seed in seeds_to_try:
            if time.time() > deadline:
                break
            rsf = make_randomized_greedy(sf, 0.1, seed)
            prize, path = run_greedy(rsf, seed)
            if len(path) > 1:
                good_paths.append((prize, path))
                
    # Sort paths by prize
    good_paths.sort(key=lambda x: x[0], reverse=True)
    
    # Construct heuristic matrix
    # Base heuristic: prize / distance (standard)
    base_H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i, j] > eps:
                base_H[i, j] = prizes[j] / distance_matrix[i, j]
            else:
                base_H[i, j] = eps
    
    # Boost edges that appear in the top K good paths
    # We want to guide the ACO towards these specific transitions
    top_k = min(5, len(good_paths))
    
    boosted_H = base_H.copy()
    
    for k in range(top_k):
        prize, path = good_paths[k]
        # Iterate through the path and boost the edges taken
        for idx in range(len(path) - 1):
            i = path[idx]
            j = path[idx + 1]
            # Boost factor: higher for better paths, higher for earlier steps?
            # Let's just add a constant boost proportional to the prize of the path
            # Or simply multiply by a factor.
            # Let's add a large value to make it very attractive
            # The ACO chooses proportional to H. If we make H[i,j] very large,
            # the ant will almost certainly choose it if it's a valid candidate.
            
            # We need to be careful not to make it infinite.
            # Let's set it to a value significantly larger than the base.
            # Base values are roughly prize/distance.
            # Max prize 1, min distance ~0.01 (in unit square with 50 pts).
            # So base H is up to ~100.
            # Let's set boosted H to 1000 + prize * 100
            boost_val = 1000 + prize * 100
            if boosted_H[i, j] < boost_val:
                boosted_H[i, j] = boost_val
    
    # Also, ensure that the "return to 0" isn't explicitly in the heuristic matrix
    # because the heuristic is only used for moving from i to j (j != 0 usually, 
    # but technically j can be 0? No, "moves to an unvisited node j". 
    # Node 0 is visited at start. So j != 0.
    # But what if the ant wants to go back to 0? It can't, because 0 is visited.
    # So H[i, 0] is irrelevant.
    
    # Evaluate the boosted heuristic
    best_score = -1
    best_h = base_H
    
    # We should also evaluate the base heuristic just in case boosting hurts
    candidates = [
        ("base", base_H),
        ("boosted", boosted_H)
    ]
    
    # Maybe try a mix?
    # Mix 1: 50% base, 50% boosted (elementwise max)
    mix1 = np.maximum(base_H, boosted_H)
    candidates.append(("mix1", mix1))
    
    # Mix 2: Base + 10 * Boosted
    mix2 = base_H + 10 * (boosted_H - base_H) # This is just boosted_H scaled? No.
    # Let's just try elementwise max and sum
    mix3 = base_H + boosted_H
    candidates.append(("mix3", mix3))
    
    for name, h in candidates:
        if time.time() > deadline:
            break
        
        # Ensure validity
        h_clean = np.where(h > 0, h, 1e-9)
        
        # Simulate
        score = _simulate_aco(prizes, distance_matrix, max_len, h_clean)
        
        if score > best_score:
            best_score = score
            best_h = h_clean
            
    return best_h
