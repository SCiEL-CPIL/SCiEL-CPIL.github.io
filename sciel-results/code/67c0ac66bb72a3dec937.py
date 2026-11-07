import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=5, n_iterations=10, rho=0.9):
    """
    Fast internal ACO simulation matching the contract's parameters.
    Used to evaluate heuristic quality during the heuristic construction phase.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    # Pre-allocate arrays for speed
    for it in range(n_iterations):
        tour_lengths = np.zeros(n_ants)
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                # Calculate probabilities
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    # Use random choice
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths[ant] = tour_len
            
            # Deposit pheromone
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        # Evaporate pheromone
        tau *= rho
        
        current_best = np.min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def _two_opt_improve(tour, dist):
    """
    Apply 2-opt improvement to a tour.
    """
    n = len(tour)
    if n < 4:
        return tour, sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                
                u1 = tour[(i - 1) % n]
                v1 = tour[i]
                u2 = tour[j]
                v2 = tour[(j + 1) % n]
                
                delta = dist[u1, u2] + dist[v1, v2] - dist[u1, v1] - dist[u2, v2]
                if delta < -1e-10:
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    
    total_len = 0.0
    for i in range(n):
        total_len += dist[tour[i], tour[(i + 1) % n]]
    return tour, total_len

def _compute_mst_edges(distance_matrix):
    """
    Compute MST edges using Prim's algorithm (O(n^2) which is fine for n=50).
    Returns a set of tuples (u, v) representing undirected edges.
    """
    n = distance_matrix.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    
    # Start from node 0
    min_dist[0] = 0.0
    in_mst[0] = True
    
    mst_edges = set()
    
    for _ in range(n - 1):
        # Find the node with the minimum distance that is not in MST
        v = -1
        min_val = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < min_val:
                min_val = min_dist[i]
                v = i
        
        if v == -1:
            break # Disconnected? Should not happen for TSP
            
        in_mst[v] = True
        if parent[v] != -1:
            mst_edges.add((parent[v], v))
            
        # Update distances
        for u in range(n):
            if not in_mst[u] and distance_matrix[v, u] < min_dist[u]:
                min_dist[u] = distance_matrix[v, u]
                parent[u] = v
                
    return mst_edges

def _compute_matching_edges(distance_matrix):
    """
    Compute maximum-weight perfect matching on complete graph.
    Edge weights are -dist^2 (maximizing weight = minimizing dist^2).
    Uses a simple greedy + local improvement heuristic for speed,
    or exact if scipy is available.
    """
    n = distance_matrix.shape[0]
    
    # Try to use scipy for exact matching if available
    try:
        from scipy.optimize import linear_sum_assignment
        # We want min cost, cost = dist^2
        cost_matrix = distance_matrix ** 2
        # linear_sum_assignment finds min cost assignment
        # But TSP matching is on undirected graph, assignment is directed bipartite.
        # For TSP, a perfect matching on the complete graph is a set of disjoint edges covering all nodes.
        # The bipartite matching (assignment) gives a set of directed edges u->v such that each u has one out and one in.
        # This forms a cycle cover, not necessarily a matching.
        # However, for the purpose of "structural constraint", we can use the edges from the cycle cover.
        # Or we can just use the nearest neighbor matching heuristic.
        
        # Let's stick to a simpler structural feature: k-nearest neighbors.
        # Or use the cycle cover edges.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        edges = set()
        for u, v in zip(row_ind, col_ind):
            if u != v:
                edges.add((u, v))
                edges.add((v, u))
        return edges
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: Greedy matching
    edges = set()
    used = np.zeros(n, dtype=bool)
    # Sort all edges by distance
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            edges_list.append((distance_matrix[i, j], i, j))
    edges_list.sort()
    
    for d, u, v in edges_list:
        if not used[u] and not used[v]:
            edges.add((u, v))
            edges.add((v, u))
            used[u] = True
            used[v] = True
            if np.sum(used) == n:
                break
    return edges

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    # 1. Generate a high-quality tour (NN + 2-Opt)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    tour = [0]
    
    for _ in range(n - 1):
        dists = safe_dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    tour, _ = _two_opt_improve(tour, safe_dist)
    
    # Create masks
    tour_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_mask[u, v] = True
        tour_mask[v, u] = True
        
    # 2. Compute MST
    mst_edges = _compute_mst_edges(safe_dist)
    mst_mask = np.zeros((n, n), dtype=bool)
    for u, v in mst_edges:
        mst_mask[u, v] = True
        mst_mask[v, u] = True
        
    # 3. Compute Matching (Cycle Cover or Greedy)
    match_edges = _compute_matching_edges(safe_dist)
    match_mask = np.zeros((n, n), dtype=bool)
    for u, v in match_edges:
        match_mask[u, v] = True
        match_mask[v, u] = True
        
    # 4. K-Nearest Neighbors Mask
    k = 3
    knn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        dists = safe_dist[i, :].copy()
        dists[i] = np.inf
        k_indices = np.argsort(dists)[:k]
        knn_mask[i, k_indices] = True
        
    # 5. Grid Search for Heuristic Parameters
    # Heuristic: H_ij = (1/d_ij^beta) * (1 + alpha_tour * I_tour) * (1 + alpha_mst * I_mst) * (1 + alpha_match * I_match) * (1 + alpha_knn * I_knn)
    
    best_heur = None
    best_score = float('inf')
    
    # Parameters to search
    alphas_tour = [0.0, 1.0, 5.0, 10.0]
    alphas_mst = [0.0, 1.0, 2.0]
    alphas_match = [0.0, 1.0, 5.0]
    alphas_knn = [0.0, 1.0]
    betas = [1.0, 2.0, 3.0]
    
    # Reduced simulation for speed: 5 ants, 10 iterations
    # Total candidates: 4 * 3 * 3 * 2 * 3 = 216.
    # 216 * 0.1s = 21.6s. Safe.
    
    for beta in betas:
        for alpha_tour in alphas_tour:
            for alpha_mst in alphas_mst:
                for alpha_match in alphas_match:
                    for alpha_knn in alphas_knn:
                        if time.time() > deadline - 5:
                            break
                            
                        # Construct heuristic
                        base_h = 1.0 / (safe_dist ** beta)
                        
                        if alpha_tour > 0:
                            base_h[tour_mask] *= (1.0 + alpha_tour)
                            
                        if alpha_mst > 0:
                            base_h[mst_mask] *= (1.0 + alpha_mst)
                            
                        if alpha_match > 0:
                            base_h[match_mask] *= (1.0 + alpha_match)
                            
                        if alpha_knn > 0:
                            base_h[knn_mask] *= (1.0 + alpha_knn)
                            
                        # Sanitize
                        base_h[~np.isfinite(base_h)] = 1e-9
                        base_h[base_h <= 0] = 1e-9
                        
                        # Evaluate
                        score = _fast_aco_simulate(safe_dist, base_h, n_ants=5, n_iterations=10)
                        
                        if score < best_score:
                            best_score = score
                            best_heur = base_h.copy()
    
    # If no candidate was evaluated, fallback
    if best_heur is None:
        best_heur = 1.0 / safe_dist
        best_heur[~np.isfinite(best_heur)] = 1e-9
        best_heur[best_heur <= 0] = 1e-9
        
    return best_heur
