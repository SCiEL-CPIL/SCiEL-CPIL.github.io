import numpy as np
import time
import random

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, seed):
    """Fast ACO simulation. Returns the best total prize collected by any ant."""
    rng = np.random.RandomState(seed)
    n = len(prizes)
    pheromone = np.ones((n, n))
    dist_to_0 = distance_matrix[:, 0]
    
    global_best = 0.0
    
    for _iter in range(30):
        best_prize_this_iter = 0.0
        
        for ant_idx in range(20):
            current = 0
            length_so_far = 0.0
            prize_so_far = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                costs = distance_matrix[current, :] + dist_to_0
                feasible_mask = (~visited) & (length_so_far + costs <= max_len)
                
                if not np.any(feasible_mask):
                    break
                
                feasible_indices = np.where(feasible_mask)[0]
                tau_vals = pheromone[current, feasible_indices]
                eta_vals = heuristic[current, feasible_indices]
                
                probs = tau_vals * eta_vals
                prob_sum = np.sum(probs)
                if prob_sum <= 0:
                    break
                probs = probs / prob_sum
                
                next_node = rng.choice(feasible_indices, p=probs)
                
                length_so_far += distance_matrix[current, next_node]
                prize_so_far += prizes[next_node]
                visited[next_node] = True
                current = next_node
            
        # Track best of iteration
        # We need to store ant prizes to find the max
        # Actually, let's just track max in the loop above for efficiency
        # Wait, the logic above doesn't store ant_prizes. Let me fix that.
        # Re-implementing inner loop to track max properly.
        
    # The above implementation was slightly flawed in tracking the max per iteration correctly 
    # without storing all ant results or updating a running max. 
    # Let's rewrite _simulate_aco to be correct and fast.
    
    return global_best

def _simulate_aco_correct(prizes, distance_matrix, max_len, heuristic, seed):
    rng = np.random.RandomState(seed)
    n = len(prizes)
    pheromone = np.ones((n, n))
    dist_to_0 = distance_matrix[:, 0]
    
    global_best = 0.0
    
    for _iter in range(30):
        best_prize_this_iter = 0.0
        
        for ant_idx in range(20):
            current = 0
            length_so_far = 0.0
            prize_so_far = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                costs = distance_matrix[current, :] + dist_to_0
                feasible_mask = (~visited) & (length_so_far + costs <= max_len)
                
                if not np.any(feasible_mask):
                    break
                
                feasible_indices = np.where(feasible_mask)[0]
                tau_vals = pheromone[current, feasible_indices]
                eta_vals = heuristic[current, feasible_indices]
                
                probs = tau_vals * eta_vals
                prob_sum = np.sum(probs)
                if prob_sum <= 0:
                    break
                probs = probs / prob_sum
                
                next_node = rng.choice(feasible_indices, p=probs)
                
                length_so_far += distance_matrix[current, next_node]
                prize_so_far += prizes[next_node]
                visited[next_node] = True
                current = next_node
            
            if prize_so_far > best_prize_this_iter:
                best_prize_this_iter = prize_so_far
        
        if best_prize_this_iter > global_best:
            global_best = best_prize_this_iter
            
        pheromone *= 0.9
        pheromone += best_prize_this_iter / 10.0
        
    return global_best

def _compute_centrality_features(distance_matrix, prizes, n):
    """
    Compute graph-theoretic centrality features for each node.
    Returns a matrix of shape (n, n) where features[i, j] is the feature value of node j
    (since heuristic eta[i, j] typically depends on the target node j's attractiveness).
    Actually, the heuristic in ACO is usually eta[i,j] = 1/d[i,j] or similar.
    Here we want to model eta[i,j] based on features of node j and possibly i.
    
    The directive suggests: "Model the heuristic as a learned linear combination of graph-theoretic centrality features... allowing the optimizer to discover global structural attractors... independent of local distance constraints."
    
    This implies eta[i, j] might be a function of node j's centrality, potentially modulated by i.
    However, standard ACO heuristics are static per edge or node.
    Let's define features for each node j:
    1. Degree Centrality: 1 / (sum of distances to all other nodes) ? Or just 1/d_ij?
       Let's use "closeness" = 1 / sum_k d[j, k]
    2. Betweenness Centrality: Approximate by counting how many shortest paths pass through j.
       Or simpler: for each pair (i, k), if d[i, j] + d[j, k] is close to d[i, k], j is on a path.
       This is O(n^3) which is fine for n=50.
    3. Prize-weighted Eigenvector Centrality:
       Solve (A - lambda I) v = 0 where A is a matrix of prizes/distances?
       Or simply: v = M @ v, where M[j, k] = prizes[k] / d[j, k].
       Iterate to convergence.
    
    We will compute these features for each node j.
    Then eta[i, j] = exp( linear_combination of features of j ) * (1 / d[i, j])?
    Or just eta[i, j] = exp( linear_combination of features of j )?
    
    The directive says "independent of local distance constraints", which is risky because distance is critical for feasibility.
    However, the heuristic guides *choice*, while feasibility is handled by the mask.
    If the heuristic ignores distance, ants might choose far nodes, but the mask prevents infeasible ones.
    So, a heuristic based on node centrality/prize is valid.
    
    Let's define the feature vector for node j:
    f_j = [
        prizes[j],
        closeness[j],
        betweenness[j],
        eigenvector[j]
    ]
    
    Then eta[i, j] = exp( w . f_j )
    This makes eta[i, j] independent of i, which is a strong assumption.
    To allow for i-dependence, we could include features of i or interaction terms.
    But let's start with node-centric features.
    
    Wait, the current program uses d[i,j] and d[j,0] in the heuristic.
    The inspiration suggests moving away from hand-crafted ratios to centrality.
    
    Let's compute:
    1. closeness[j] = 1 / sum_{k != j} d[j, k]
    2. betweenness[j]: 
       For each pair (u, v) with u != v != j:
       If d[u, j] + d[j, v] == d[u, v] (within epsilon), j lies on a shortest path.
       Count such pairs. Normalize by total pairs.
    3. eigenvector[j]:
       Start with v = prizes / sum(prizes)
       Iterate: v_new[j] = sum_k (prizes[k] / d[j, k]) * v[k] / sum(...)
       Do 10 iterations.
    """
    
    # 1. Closeness
    sum_dist = np.sum(distance_matrix, axis=1)
    # Avoid division by zero for self, but d[j,j]=0
    closeness = np.zeros(n)
    mask_nonzero = sum_dist > 0
    closeness[mask_nonzero] = 1.0 / sum_dist[mask_nonzero]
    
    # 2. Betweenness (approximate)
    betweenness = np.zeros(n)
    # This is O(n^3) = 125,000 ops for n=50, very fast.
    for u in range(n):
        for v in range(n):
            if u == v: continue
            # Check all potential intermediates j
            # d[u, j] + d[j, v] == d[u, v]
            # Vectorized:
            path_costs = distance_matrix[u, :] + distance_matrix[:, v]
            shortest = distance_matrix[u, v]
            # Find j where path_costs[j] is close to shortest
            # Use a small tolerance
            on_path = np.where(np.abs(path_costs - shortest) < 1e-6)[0]
            # Exclude u and v themselves? Usually betweenness excludes endpoints.
            # If j == u or j == v, it's trivially on the path.
            for j in on_path:
                if j != u and j != v:
                    betweenness[j] += 1
    
    # Normalize betweenness
    max_between = np.max(betweenness)
    if max_between > 0:
        betweenness = betweenness / max_between
    
    # 3. Prize-weighted Eigenvector Centrality
    # v = M @ v
    # M[j, k] = prizes[k] / d[j, k] if j != k else 0
    # Normalize v at each step
    eigenvector = prizes.copy() / (np.sum(prizes) + 1e-9)
    for _ in range(10):
        # Compute new eigenvector
        # new_v[j] = sum_k M[j, k] * v[k]
        # M[j, k] = prizes[k] / d[j, k]
        # We can compute this as: (prizes / d) @ v
        # d is distance_matrix
        # Create matrix W where W[j, k] = prizes[k] / d[j, k]
        # To avoid division by zero, set d[j, j] to 1 or large number
        d_safe = distance_matrix.copy()
        np.fill_diagonal(d_safe, 1.0)
        W = prizes[None, :] / d_safe
        new_v = W @ eigenvector
        norm = np.sum(new_v)
        if norm > 0:
            eigenvector = new_v / norm
        else:
            break
    
    # Features matrix: shape (n, 4)
    # Each row j has [prize, closeness, betweenness, eigenvector]
    features = np.column_stack([
        prizes,
        closeness,
        betweenness,
        eigenvector
    ])
    
    return features

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Compute centrality features
    features = _compute_centrality_features(distance_matrix, prizes, n)
    
    # We want to find weights w such that eta[i, j] = exp(w . features[j])
    # Or maybe eta[i, j] = exp(w . features[j]) * (1 / d[i, j])^alpha?
    # The directive says "independent of local distance constraints", so let's try pure centrality first.
    # But pure centrality might be bad if it ignores distance completely.
    # Let's include a distance term as a feature too?
    # Feature 4: 1/d[i,j] is not a node feature, it's an edge feature.
    # If we stick to node features, eta is symmetric in a sense (depends only on j).
    # eta[i, j] = eta[j]
    
    # Let's search for weights w in R^4.
    # Also, we might want to scale the exponent.
    # eta = exp(scale * (w . features))
    
    # Hyperparameters: w (4 values) and scale (1 value).
    # This is a 5-dimensional search space.
    # We can use a simple grid or random search.
    # Given n=50 and 30 ACO iterations, simulation is fast.
    # Let's do a random search with a few hundred iterations.
    
    best_score = -1
    best_w = np.array([1.0, 0.5, 0.5, 1.0]) # Initial guess
    best_scale = 1.0
    
    # Time budget: 280s
    time_limit = 280.0
    
    # Random search
    np.random.seed(42)
    
    # We will test 200 random weight vectors
    # To speed up, we can use a small number of seeds for evaluation during search
    eval_seeds = [1, 2]
    
    for iter_search in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random weights
        # Weights can be positive or negative
        # Scale is positive
        w = np.random.normal(0, 1, 4)
        scale = np.random.uniform(0.5, 3.0)
        
        # Compute heuristic
        # eta[i, j] = exp(scale * (w . features[j]))
        # features is (n, 4), so w . features[j] is (n,)
        scores_linear = w @ features.T  # shape (n,)
        eta_values = np.exp(scale * scores_linear)
        
        # eta is (n, 1), broadcast to (n, n)
        eta = np.tile(eta_values, (n, 1))
        
        # Ensure positive and finite
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        # Evaluate
        try:
            scores = []
            for seed in eval_seeds:
                score = _simulate_aco_correct(prizes, distance_matrix, max_len, eta, seed)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_w = w.copy()
                best_scale = scale
                
        except Exception:
            continue
            
    # Final heuristic generation
    scores_linear = best_w @ features.T
    eta_values = np.exp(best_scale * scores_linear)
    eta = np.tile(eta_values, (n, 1))
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Safety check
    if not np.all(np.isfinite(eta)):
        eta = np.ones_like(distance_matrix)
        
    return eta
